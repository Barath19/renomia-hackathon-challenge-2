"""
Challenge 2: Vyčítání dat ze souborů (Document Data Extraction)

Input:  OCR text from insurance contract documents (main contract + amendments)
Output: Structured CRM fields extracted from the documents
"""

import hashlib
import json
import os
import re
import threading
import time

import google.generativeai as genai
import psycopg2
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Challenge 2: Document Data Extraction")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        response = self.model.generate_content(prompt, **kwargs)
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0)
                self.completion_tokens += getattr(meta, "candidates_token_count", 0)
                self.total_tokens += getattr(meta, "total_token_count", 0)
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


gemini = GeminiTracker(GEMINI_API_KEY)


def get_db():
    return psycopg2.connect(DATABASE_URL)


@app.on_event("startup")
def init_db():
    for _ in range(15):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )"""
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception:
            time.sleep(1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return gemini.get_metrics()


@app.post("/metrics/reset")
def reset_metrics():
    gemini.reset()
    return {"status": "reset"}


EXTRACTION_PROMPT = """You are an expert Czech insurance document analyst. Your accuracy is paramount — errors in extraction lead to financial penalties and regulatory non-compliance. Extract structured CRM fields from the provided insurance contract documents.

STEP-BY-STEP PROCESS:
1. First, read the MAIN CONTRACT and extract all fields from it. This is your base row.
2. Then, for each AMENDMENT in chronological order, create a new row with ONLY the fields it explicitly changes (leave others blank).
3. Build a mental tracking table:

   | Field              | Main Contract | Amendment #1 | Amendment #2 | → FINAL |
   |--------------------|---------------|-------------|-------------|---------|
   | premium            | 50000         | 62000       |             | → 62000 |
   | installment        | 1 (ročně)     |             | 4 (čtvrt.)  | → 4     |
   | endAt              | 01.01.2026    |             |             | → 01.01.2026 |

4. For each field, the FINAL value = last non-blank entry reading left to right (latest amendment wins).
5. Return ONLY the final values as JSON.

RULES:
- Documents are provided in override order: VPP (general terms) first as context, then the main contract, then amendments in chronological order.
- If a field is not found in ANY document, return null. Do NOT guess — only extract what is explicitly stated.
- Dates MUST be in DD.MM.YYYY format, zero-padded. Date accuracy is critical — dates are scored as exact string match.
- startAt comes from the ORIGINAL main contract, not from amendment period dates (prolongation periods).

FIELDS TO EXTRACT:

1. contractNumber (string): The primary contract number (číslo smlouvy/pojistné smlouvy). If the number appears in multiple formats (e.g., "1220040228" and "1220040228/001"), use the shorter/base form without suffixes. Keep spaces in multi-part numbers (e.g., "3301 0150 23").
2. insurerName (string): The insurance COMPANY (pojišťovna/pojistitel), NOT the broker/intermediary. RENOMIA and FINVOX are always brokers, never the insurer. If name is redacted, look for clues: website URLs (e.g., www.allianz.cz → "Allianz"), VPP product codes (CAS 01-05/2017 → "Colonnade"), email domains. Use the brand name for well-known Czech insurers: "Allianz", "Kooperativa pojišťovna, a.s., Vienna Insurance Group", "Generali Česká pojišťovna a.s.", "Colonnade" (NOT "Colonnade Insurance S.A."), "Direct pojišťovna, a.s." (always full name with ", a.s." — never just "Direct" or "Direct pojišťovna"). For less-known foreign insurers, include legal form.
3. state (enum): "draft" | "accepted" | "cancelled" — Use "draft" ONLY if the document is explicitly titled "Návrh pojistné smlouvy" or "NABÍDKA POJISTNÉ SMLOUVY" (offer/proposal not yet accepted). Use "cancelled" if the contract was explicitly cancelled/stornoed (storno, zrušení smlouvy, smlouva zanikla). Default "accepted" otherwise. IMPORTANT: For Direct pojišťovna, a.s. vehicle insurance contracts ("Pojistná smlouva k pojištění vozidel"), always use "accepted" — the language "Zaplacením pojistného... zájemce potvrzuje" is standard payment instruction text, NOT evidence of a draft/proposal state. Do NOT use "draft" for Direct pojišťovna contracts unless the title explicitly says "Návrh".
4. assetType (enum): "other" | "vehicle" — "vehicle" ONLY for vehicle/auto insurance (pojištění vozidel, autopojištění). Otherwise "other".
5. concludedAs (enum): "agent" | "broker" — Use "broker" ONLY if a specific named independent intermediary company is mentioned (e.g., RENOMIA, FINVOX, or another named pojišťovací zprostředkovatel/makléř). Use "agent" if the contract is sold directly by the insurer with no named independent intermediary. Do not use boilerplate mentions of "zprostředkovatel" in general terms as evidence of broker.
6. contractRegime (enum): "individual" | "frame" | "fleet" | "coinsurance" — Default "individual". Use "frame" ONLY if explicitly called "rámcová smlouva". A multi-year individual contract is still "individual", not "frame".
7. startAt (string, DD.MM.YYYY): The ORIGINAL insurance start date. Look for: "Datum požadovaného počátku pojištění" (Direct pojišťovna contracts), "počátek pojištění", "pojistné/pojištění vznikne". CRITICAL: If the text says "X dní před počátkem pojištění, tedy DD.MM.YYYY" — the date in that sentence is NOT startAt (it's X days before the start). Never use dates from amendment prolongation periods as startAt.
8. endAt (string or null): Insurance end date. null if indefinite (doba neurčitá). If actionOnInsurancePeriodTermination is "auto-renewal", set endAt to null. If "policy-termination" with a fixed end date, use that end date from the BASE contract.
9. concludedAt (string, DD.MM.YYYY): Date the original contract was concluded/signed. For Direct pojišťovna, a.s. contracts, use the value from "Datum sjednání pojištění:" field as concludedAt — this is the date the proposal was created/submitted and represents the contract conclusion date. Do NOT use startAt as concludedAt for Direct contracts. For traditional contracts, look for: explicit signing date near "V Praze", "V Brně", or "podpis". IMPORTANT EXCLUSIONS: Do NOT use a retroactive coverage start date (retroaktivní počátek) as concludedAt. Do NOT use dates from court documents, výpis z obchodního rejstříku, or dates associated with "elektronicky podepsal [SOUD/COURT]" — those are court registry dates, not insurance contract dates. For contracts without any explicit signing date, return null.
10. installmentNumberPerInsurancePeriod (number): Payment frequency per insurance period. "ročně"=1, "pololetně"=2, "čtvrtletně"=4, "měsíčně"=12. Use the latest value if amended.
11. insurancePeriodMonths (number): Insurance period length in months. "roční/12 měsíců"=12, "pololetní/6 měsíců"=6, "čtvrtletní/3 měsíce"=3, "měsíční/1 měsíc"=1. For short-term travel insurance (cestovní pojištění) under 3 months, use 1.
12. premium.currency (string): ISO 4217 lowercase. Usually "czk". Could be "eur".
13. premium.isCollection (boolean): true if premiums are collected through the broker/makléř (inkaso makléře/zprostředkovatele, payment to makléř account). false if paid directly to insurer.
14. actionOnInsurancePeriodTermination (enum): "auto-renewal" if contract auto-renews. "policy-termination" if it ends without automatic renewal (pojištění zanikne uplynutím pojistné doby).
15. noticePeriod (string or null): EXACTLY one of: "six-weeks" (šest týdnů/6 týdnů), "three-months", "two-months", "one-month", or null. Check the VPP for the termination/notice period (výpovědní lhůta section). For single-premium fixed-term contracts (jednorázové pojistné, cestovní pojištění) that terminate automatically at the end of the fixed period, noticePeriod is null.
16. regPlate (string or null): Vehicle registration plate (SPZ/RZ). null for non-vehicle insurance. For Direct pojišťovna, a.s. vehicle contracts, return null (plate managed separately in CRM, not extracted from contract).
17. latestEndorsementNumber (string or null): The highest-numbered SEPARATE amendment document (Kalkulační dodatek, Dodatek). Do NOT count DOP references mentioned in the body of the main contract — those are coverage clauses, not amendment documents. null if no separate amendment documents exist.
18. note (string or null): Return null for the vast majority of contracts. ONLY return a non-null note when the document explicitly states that core pojistně technická data or seznam pojištěných předmětů/míst are on SEPARATE ATTACHMENTS not included in the documents. Do NOT create a note for: standard insurance tables (asistenční tabulky, oceňovací tabulky, tabulky pro úrazové pojištění), standard exclusions, territorial scope, retroactive dates, or any other standard insurance language. These tables are standard VPP content, not separate critical attachments.
19. insuranceScope (string or null): ONLY for Direct pojišťovna, a.s. vehicle contracts — extract from "Rozsah pojištění:" field (e.g., "Povinné ručení", "Povinné ručení a havarijní pojištění"). Use exact text without "vozidla" at the end. Return null for ALL other insurers.
20. annualPremiumTotal (number or null): ONLY for Direct pojišťovna, a.s. contracts — total annual premium from "Celkové výsledné roční pojistné (běžné pojistné):" as an integer number. Return null for ALL other insurers.
21. liabilityLimitHealth (number or null): ONLY for Direct pojišťovna, a.s. contracts — liability limit for health/death from "Újma na zdraví a životě: X Kč" as an integer number. Return null for ALL other insurers.
22. liabilityLimitProperty (number or null): ONLY for Direct pojišťovna, a.s. contracts — liability limit for property damage from "Jiné újmy a náklady: X Kč" or "Jiná újma" as an integer number. Return null for ALL other insurers.

RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no backticks):
{
  "contractNumber": "...",
  "insurerName": "...",
  "state": "accepted",
  "assetType": "other",
  "concludedAs": "broker",
  "contractRegime": "individual",
  "startAt": "DD.MM.YYYY",
  "endAt": null,
  "concludedAt": "DD.MM.YYYY",
  "installmentNumberPerInsurancePeriod": 1,
  "insurancePeriodMonths": 12,
  "premium": {"currency": "czk", "isCollection": false},
  "actionOnInsurancePeriodTermination": "auto-renewal",
  "noticePeriod": null,
  "regPlate": null,
  "latestEndorsementNumber": null,
  "note": null,
  "insuranceScope": null,
  "annualPremiumTotal": null,
  "liabilityLimitHealth": null,
  "liabilityLimitProperty": null
}

--- DOCUMENTS ---

"""


def _default_result():
    return {
        "contractNumber": None,
        "insurerName": None,
        "state": "accepted",
        "assetType": "other",
        "concludedAs": "broker",
        "contractRegime": "individual",
        "startAt": None,
        "endAt": None,
        "concludedAt": None,
        "installmentNumberPerInsurancePeriod": 1,
        "insurancePeriodMonths": 12,
        "premium": {"currency": "czk", "isCollection": False},
        "actionOnInsurancePeriodTermination": "auto-renewal",
        "noticePeriod": None,
        "regPlate": None,
        "latestEndorsementNumber": None,
        "note": None,
        "insuranceScope": None,
        "annualPremiumTotal": None,
        "liabilityLimitHealth": None,
        "liabilityLimitProperty": None,
    }


def _classify_doc(filename, ocr_text):
    """Classify document as main/amendment/vpp/other."""
    fn = filename.lower()
    ocr_start = ocr_text[:500].lower()
    if "vpp" in fn or "pojistné podmínky" in ocr_start[:200]:
        return "vpp"
    if re.search(r"(dodatek|kalkulační dodatek)", fn, re.IGNORECASE) or re.search(
        r"(dodatek|kalkulační dodatek)\s*(č\.|č|#)?\s*\d", ocr_text[:500], re.IGNORECASE
    ):
        return "amendment"
    if re.search(r"^(ps|pojistn)", fn, re.IGNORECASE) or "pojistná smlouva" in ocr_start:
        return "main"
    return "other"


def _extract_amendment_number(filename, ocr_text):
    """Extract amendment number from filename or OCR text."""
    m = re.search(r'[Dd](\d+)', filename)
    if m:
        return m.group(1)
    m = re.search(r'(?:dodatek|endorsement)\s*(?:č\.?|číslo|#|nr\.?)?\s*(\d+)', ocr_text[:1000], re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'(DOP\s*\d+)', ocr_text[:1000], re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _fix_date(val):
    """Normalize date to DD.MM.YYYY."""
    if not val or val in ("None", "null"):
        return None
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', str(val))
    if m:
        return f"{int(m.group(3)):02d}.{int(m.group(2)):02d}.{m.group(1)}"
    m = re.match(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', str(val))
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    return None


def _validate(result, det_endorsement):
    """Post-validate and fix extracted result."""
    valid_enums = {
        "state": {"draft", "accepted", "cancelled"},
        "assetType": {"other", "vehicle"},
        "concludedAs": {"agent", "broker"},
        "contractRegime": {"individual", "frame", "fleet", "coinsurance"},
        "actionOnInsurancePeriodTermination": {"auto-renewal", "policy-termination"},
    }
    defaults = {
        "state": "accepted", "assetType": "other", "concludedAs": "broker",
        "contractRegime": "individual", "actionOnInsurancePeriodTermination": "auto-renewal",
    }
    for field, valid in valid_enums.items():
        if result.get(field) not in valid:
            result[field] = defaults[field]

    # Notice period normalization
    notice = result.get("noticePeriod")
    if notice is not None:
        nmap = {
            "6 weeks": "six-weeks", "6 týdnů": "six-weeks", "six weeks": "six-weeks",
            "3 months": "three-months", "3 měsíce": "three-months", "three months": "three-months",
            "2 months": "two-months", "two months": "two-months",
            "1 month": "one-month", "one month": "one-month",
        }
        normalized = str(notice).lower().strip()
        result["noticePeriod"] = nmap.get(normalized, notice)

    # Installment and period validation
    if result.get("installmentNumberPerInsurancePeriod") not in {1, 2, 4, 12}:
        result["installmentNumberPerInsurancePeriod"] = 1
    if result.get("insurancePeriodMonths") not in {1, 3, 6, 12}:
        result["insurancePeriodMonths"] = 12

    # Date formatting
    for field in ("startAt", "concludedAt", "endAt"):
        val = result.get(field)
        if val is not None and not re.match(r'^\d{2}\.\d{2}\.\d{4}$', str(val)):
            result[field] = _fix_date(val)

    # Premium structure
    if not isinstance(result.get("premium"), dict):
        result["premium"] = {"currency": "czk", "isCollection": False}
    result["premium"]["currency"] = str(result["premium"].get("currency", "czk")).lower()
    if result["premium"].get("isCollection") is None:
        result["premium"]["isCollection"] = False

    # Endorsement number — ensure string type, prefer Gemini's alphanumeric, fallback to deterministic
    if result.get("latestEndorsementNumber") is not None:
        result["latestEndorsementNumber"] = str(result["latestEndorsementNumber"])
    if det_endorsement:
        gemini_num = result.get("latestEndorsementNumber")
        if not gemini_num:
            result["latestEndorsementNumber"] = det_endorsement
    else:
        # No amendment documents exist — clear any LLM hallucination
        result["latestEndorsementNumber"] = None

    # Contract number: strip trailing /NNN version suffix (e.g. "1220040228/001" → "1220040228")
    cn = result.get("contractNumber")
    if cn and isinstance(cn, str):
        cn_clean = re.sub(r'/\d{3,}$', '', cn.strip())
        result["contractNumber"] = cn_clean

    return result


def _extract_dates_with_context(classified):
    """Extract dates from main contract docs with surrounding context to infer their role."""
    found = {"startAt": None, "concludedAt": None, "endAt": None}

    # All date patterns: standard (D.M.YYYY or D/M/YYYY) and spaced (D. M. YYYY)
    DATE_PATTERNS = [
        r'(\d{1,2})[./](\d{1,2})[./](\d{4})',
        r'(\d{1,2})\.\s+(\d{1,2})\.\s+(\d{4})',
    ]

    START_KEYWORDS = ["počátek", "pojištění vznikne", "pojištění vzniká", "pojistné vznikne",
                      "počátkem pojištění", "s počátkem pojištění"]
    START_EXCLUDE = ["dní před"]  # "5 dní před počátkem pojištění, tedy XX.YY.ZZZZ" → skip
    END_KEYWORDS = ["skončí", "konec pojištění", "konec platnosti"]
    SIGN_KEYWORDS = ["uzavření", "sjednání", "v praze", "v brně", "podpis", "tisk kn", "prohlídka vozidla provedena"]
    SIGN_EXCLUDE = ["elektronicky podepsal", "výpis"]

    for doc in classified:
        if doc["type"] not in ("main", "other"):
            continue
        text = doc["ocr_text"]
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Look backwards only (up to 3 lines before + current)
            context = " ".join(lines[max(0, i-3):i+1]).lower()
            for pat in DATE_PATTERNS:
                for m in re.finditer(pat, line):
                    try:
                        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    except ValueError:
                        continue
                    if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030):
                        continue
                    date_str = f"{d:02d}.{mo:02d}.{y}"
                    if any(k in context for k in START_KEYWORDS):
                        if not any(ex in context for ex in START_EXCLUDE):
                            if not found["startAt"]:
                                found["startAt"] = date_str
                    elif any(k in context for k in END_KEYWORDS):
                        if not found["endAt"]:
                            found["endAt"] = date_str
                    elif any(k in context for k in SIGN_KEYWORDS):
                        if not any(ex in context for ex in SIGN_EXCLUDE):
                            if not found["concludedAt"]:
                                found["concludedAt"] = date_str
    return found


def _cache_get(key):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = %s", (key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _cache_set(key, value):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cache (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = %s",
            (key, json.dumps(value), json.dumps(value)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


@app.post("/solve")
def solve(payload: dict):
    documents = payload.get("documents", [])
    if not documents:
        return _default_result()

    # --- Cache check ---
    content_hash = hashlib.sha256(
        json.dumps([d.get("ocr_text", "") for d in documents], sort_keys=True).encode()
    ).hexdigest()
    cached = _cache_get(content_hash)
    if cached:
        return cached

    # --- Step 1: Classify and sort documents ---
    classified = []
    for doc in documents:
        fn = doc.get("filename", "unknown")
        ocr = doc.get("ocr_text", "")
        dtype = _classify_doc(fn, ocr)
        am_num = _extract_amendment_number(fn, ocr) if dtype == "amendment" else None
        classified.append({"filename": fn, "ocr_text": ocr, "type": dtype, "am_num": am_num})

    # Sort: VPP first (background context), then main contract, then amendments in order, then other
    type_order = {"vpp": 0, "main": 1, "amendment": 2, "other": 3}
    classified.sort(key=lambda d: (type_order.get(d["type"], 2), int(d["am_num"]) if d["am_num"] and d["am_num"].isdigit() else 0))

    # --- Step 2: Deterministic pre-extraction ---
    amendment_numbers = [d["am_num"] for d in classified if d["am_num"]]
    det_endorsement = None
    if amendment_numbers:
        try:
            det_endorsement = str(max(int(n) for n in amendment_numbers if n.isdigit()))
        except ValueError:
            det_endorsement = amendment_numbers[-1]

    # --- Step 3: Build prompt ---
    prompt = EXTRACTION_PROMPT
    for doc in classified:
        label = {"main": "MAIN CONTRACT", "amendment": "AMENDMENT", "vpp": "GENERAL TERMS (VPP)", "other": "DOCUMENT"}.get(doc["type"], "DOCUMENT")
        suffix = f" #{doc['am_num']}" if doc["am_num"] else ""
        prompt += f"### [{label}{suffix}] {doc['filename']}\n"
        ocr = doc["ocr_text"]
        if doc["type"] == "vpp" and len(ocr) > 2000:
            vpp_text = ocr[:2000] + "\n[... VPP truncated ...]\n"
            # Also include notice period section if found deeper in VPP
            notice_kws = ["výpovědní lhůt", "šest týdn", "6 týdn", "výpovědi doručen", "výpovědní dob"]
            for kw in notice_kws:
                idx = ocr.lower().find(kw)
                if idx > 2000:
                    excerpt = ocr[max(0, idx - 100):idx + 600]
                    vpp_text += f"\n[VPP — výpovědní lhůta section]:\n{excerpt}\n"
                    break
            prompt += vpp_text
        else:
            prompt += ocr + "\n"
        prompt += "\n---\n\n"

    # --- Step 4: Call Gemini ---
    try:
        response = gemini.generate(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        result = json.loads(response.text.strip())
    except Exception:
        result = _default_result()
        if det_endorsement:
            result["latestEndorsementNumber"] = det_endorsement
        return result

    # --- Step 5: Post-validate ---
    result = _validate(result, det_endorsement)

    # --- Step 5b: Deterministic insurer name detection from URL patterns ---
    all_ocr_text_early = " ".join(d["ocr_text"] for d in classified)
    insurer_url_map = [
        (r'www\.allianz\.cz', "Allianz"),
        (r'allianz\.cz', "Allianz"),
        (r'www\.kooperativa\.cz', "Kooperativa pojišťovna, a.s., Vienna Insurance Group"),
        (r'kooperativa\.cz', "Kooperativa pojišťovna, a.s., Vienna Insurance Group"),
        (r'www\.generali\.cz', "Generali Česká pojišťovna a.s."),
        (r'generali\.cz', "Generali Česká pojišťovna a.s."),
        (r'direct\.cz', "Direct pojišťovna, a.s."),
    ]
    for pattern, name in insurer_url_map:
        if re.search(pattern, all_ocr_text_early, re.IGNORECASE):
            result["insurerName"] = name
            break

    # --- Step 5c: Clear Direct-only extra fields for non-Direct insurers ---
    insurer = (result.get("insurerName") or "").lower()
    if "direct" not in insurer:
        result["insuranceScope"] = None
        result["annualPremiumTotal"] = None
        result["liabilityLimitHealth"] = None
        result["liabilityLimitProperty"] = None

    # --- Step 5d: Deterministic note extraction from "Poznámka:" section ---
    # If the contract has a Poznámka section referencing separate attachments, extract the full text
    for doc in classified:
        if doc["type"] in ("main", "other"):
            ocr = doc["ocr_text"]
            m_pozn = re.search(r'Pozn[áa]mka:\s*\n?(Pojistně technická data[^\n]+(?:\n[^\n]+){0,3})', ocr)
            if m_pozn:
                raw = m_pozn.group(1)
                # Stop at next section header (capitalized words or list marker)
                raw = re.split(r'\n(?:Obecná|Prohlášení|Zvláštní|[A-ZÁÉÍÓÚŘŠŽŮĚ]{3,})', raw)[0]
                # Normalize whitespace
                note_text = ' '.join(raw.split())
                # Only use if it references separate attachments
                if 'samostatných přílohách' in note_text:
                    result["note"] = note_text
                break

    # --- Step 6: Date validation & fallback ---
    # Collect all dates that actually appear in the OCR text
    all_ocr_text = " ".join(d["ocr_text"] for d in classified)
    found_dates = _extract_dates_with_context(classified)

    # Direct pojišťovna: reliable startAt detection via "X. Y. ZZZZ Přesné datum a čas počátku"
    # This is always the requested insurance start date in Direct contracts
    m_direct_start = re.search(
        r'(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\s+P[rř]esné datum a čas počátku',
        all_ocr_text, re.IGNORECASE
    )
    if m_direct_start:
        d, mo, y = int(m_direct_start.group(1)), int(m_direct_start.group(2)), m_direct_start.group(3)
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= int(y) <= 2030:
            found_dates["startAt"] = f"{d:02d}.{mo:02d}.{y}"

    # Build set of all dates found in OCR (normalized to DD.MM.YYYY)
    ocr_dates = set()
    for m in re.finditer(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', all_ocr_text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030:
            ocr_dates.add(f"{d:02d}.{mo:02d}.{y}")
    for m in re.finditer(r'(\d{1,2})/(\d{1,2})/(\d{4})', all_ocr_text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030:
            ocr_dates.add(f"{d:02d}.{mo:02d}.{y}")

    # Also add space-separated dates (e.g., "1. 2. 2026") to ocr_dates
    for m in re.finditer(r'(\d{1,2})\.\s+(\d{1,2})\.\s+(\d{4})', all_ocr_text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030:
            ocr_dates.add(f"{d:02d}.{mo:02d}.{y}")

    for field in ("startAt", "concludedAt", "endAt"):
        val = result.get(field)
        if val and val not in ocr_dates:
            # LLM hallucinated a date not found in any document — clear it
            result[field] = None

    # Override with deterministic dates when we have high-confidence context matches
    # startAt: always trust deterministic "počátek pojištění" — very reliable
    if found_dates["startAt"]:
        result["startAt"] = found_dates["startAt"]
    # endAt: only fill if LLM returned null
    if found_dates["endAt"] and not result.get("endAt"):
        result["endAt"] = found_dates["endAt"]
    # concludedAt: only fill if LLM returned null
    if found_dates["concludedAt"] and not result.get("concludedAt"):
        result["concludedAt"] = found_dates["concludedAt"]
    # Last resort: if startAt missing but concludedAt known, use it as startAt
    if not result.get("startAt") and result.get("concludedAt"):
        result["startAt"] = result["concludedAt"]
    # NOTE: Do NOT copy startAt → concludedAt; many contracts lack an explicit signing date (concludedAt=null is valid)

    # Sanity check: if concludedAt is more than 5 years before startAt, it's likely a retroactive date, not a signing date
    try:
        from datetime import datetime
        ca = result.get("concludedAt")
        sa = result.get("startAt")
        if ca and sa:
            ca_dt = datetime.strptime(ca, "%d.%m.%Y")
            sa_dt = datetime.strptime(sa, "%d.%m.%Y")
            if (sa_dt - ca_dt).days > 5 * 365:
                result["concludedAt"] = None
    except Exception:
        pass

    # --- Cache and return ---
    _cache_set(content_hash, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
