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
1. First, read the MAIN CONTRACT and extract all 17 fields from it. This is your base.
2. Then, for each AMENDMENT in order, identify ONLY the fields it explicitly changes. Apply those changes to your base values.
3. The final result is the base values with all amendment overrides applied.

RULES:
- Documents are provided in override order: VPP (general terms) first as context, then the main contract, then amendments in chronological order.
- If a field is not found in ANY document, return null. Do NOT guess — only extract what is explicitly stated.
- Dates MUST be in DD.MM.YYYY format, zero-padded. Date accuracy is critical — dates are scored as exact string match.
- startAt and concludedAt come from the ORIGINAL main contract signing, not from amendment dates.

FIELDS TO EXTRACT:

1. contractNumber (string): The contract number (číslo smlouvy/pojistné smlouvy). Keep the exact format including spaces.
2. insurerName (string): The insurance COMPANY (pojišťovna/pojistitel), NOT the broker/intermediary. RENOMIA is always a broker, never the insurer. If the name is redacted, look for clues: website URLs (e.g., www.allianz.cz → "Allianz"), VPP product codes (e.g., CAS 01-05/2017 → Colonnade), email domains. For international insurers include legal form (e.g., "Colonnade Insurance S.A."). For Czech insurers use the brand name (e.g., "Allianz", "Kooperativa").
3. state (enum): "draft" | "accepted" | "cancelled" — Default "accepted" unless explicitly stated otherwise.
4. assetType (enum): "other" | "vehicle" — "vehicle" ONLY for vehicle/auto insurance (pojištění vozidel, autopojištění). Otherwise "other".
5. concludedAs (enum): "agent" | "broker" — Default "broker" for Renomia contracts. Use "broker" if any makléř/zprostředkovatel is mentioned.
6. contractRegime (enum): "individual" | "frame" | "fleet" | "coinsurance" — Default "individual". Use "frame" ONLY if explicitly called "rámcová smlouva". A multi-year individual contract is still "individual", not "frame".
7. startAt (string, DD.MM.YYYY): The original insurance start date (počátek pojištění) from the main contract. This is when the insurance FIRST began, not a later period start.
8. endAt (string or null): Insurance end date (konec pojištění). null if indefinite (doba neurčitá). If actionOnInsurancePeriodTermination is "auto-renewal", set endAt to null. If "policy-termination" with a fixed end date, use that date.
9. concludedAt (string, DD.MM.YYYY): Date the ORIGINAL contract was concluded/signed (datum uzavření/sjednání). This is from the main contract, not from amendments.
10. installmentNumberPerInsurancePeriod (number): Payment frequency per insurance period. "ročně"=1, "pololetně"=2, "čtvrtletně"=4, "měsíčně"=12. Use the latest value if amended.
11. insurancePeriodMonths (number): Insurance period length in months. "roční/12 měsíců"=12, "pololetní/6 měsíců"=6, "čtvrtletní/3 měsíce"=3, "měsíční/1 měsíc"=1. Use the latest value if amended.
12. premium.currency (string): ISO 4217 lowercase. Usually "czk". Could be "eur".
13. premium.isCollection (boolean): true if premiums are collected through the broker/makléř (inkaso makléře/zprostředkovatele, payment to makléř account). false if paid directly to insurer.
14. actionOnInsurancePeriodTermination (enum): "auto-renewal" if contract auto-renews (automatické prodloužení/prolongace, "prodlužuje se automaticky"). "policy-termination" if it ends after the fixed term without automatic renewal (pojištění zanikne uplynutím pojistné doby, renewed only by numbered dodatek).
15. noticePeriod (string or null): EXACTLY one of: "six-weeks", "three-months", "two-months", "one-month", or null. null if no notice period applies.
16. regPlate (string or null): Vehicle registration plate (SPZ/RZ). null for non-vehicle insurance.
17. latestEndorsementNumber (string or null): The highest amendment/endorsement number found across ALL documents. Could be a simple number like "3" or alphanumeric like "DOP 098". Includes both separate amendment documents and embedded DOP clauses. null if no amendments exist.
18. note (string or null): Concise summary (1-3 sentences) of key special conditions, exclusions, territorial scope, and notable coverage details. null if nothing notable.

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
  "note": null
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

    return result


def _extract_dates_with_context(classified):
    """Extract dates from main contract docs with surrounding context to infer their role."""
    found = {"startAt": None, "concludedAt": None, "endAt": None}
    for doc in classified:
        if doc["type"] not in ("main", "other"):
            continue
        text = doc["ocr_text"]
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Get surrounding lines for context
            context = " ".join(lines[max(0, i-3):i+2]).lower()
            # Find dates in this line
            for m in re.finditer(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', line):
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030):
                    continue
                date_str = f"{d:02d}.{mo:02d}.{y}"
                # Infer role from context
                if any(k in context for k in ["počátek", "pojištění vznikne", "pojištění vzniká"]):
                    if not found["startAt"]:
                        found["startAt"] = date_str
                elif any(k in context for k in ["skončí", "konec pojištění", "konec platnosti"]):
                    if not found["endAt"]:
                        found["endAt"] = date_str
                elif any(k in context for k in ["uzavření", "sjednání", "v praze", "v brně", "podpis"]):
                    # Exclude electronic signatures / court excerpts
                    if "elektronicky podepsal" not in context and "výpis" not in context:
                        if not found["concludedAt"]:
                            found["concludedAt"] = date_str
            # Also match DD/MM/YYYY format
            for m in re.finditer(r'(\d{1,2})/(\d{1,2})/(\d{4})', line):
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2030):
                    continue
                date_str = f"{d:02d}.{mo:02d}.{y}"
                if any(k in context for k in ["počátek", "pojištění vznikne", "pojištění vzniká"]):
                    if not found["startAt"]:
                        found["startAt"] = date_str
                elif any(k in context for k in ["skončí", "konec pojištění"]):
                    if not found["endAt"]:
                        found["endAt"] = date_str
                elif any(k in context for k in ["uzavření", "sjednání", "v praze", "v brně", "podpis"]):
                    # Exclude electronic signatures / court excerpts
                    if "elektronicky podepsal" not in context and "výpis" not in context:
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
            prompt += ocr[:2000] + "\n[... VPP truncated ...]\n"
        else:
            prompt += ocr + "\n"
        prompt += "\n---\n\n"

    # --- Step 4: Call Gemini ---
    try:
        response = gemini.generate(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
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

    # --- Step 6: Date validation & fallback ---
    # Collect all dates that actually appear in the OCR text
    all_ocr_text = " ".join(d["ocr_text"] for d in classified)
    found_dates = _extract_dates_with_context(classified)

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
    # Last resort: if one is missing, copy from the other
    if not result.get("startAt") and result.get("concludedAt"):
        result["startAt"] = result["concludedAt"]
    if not result.get("concludedAt") and result.get("startAt"):
        result["concludedAt"] = result["startAt"]

    # --- Cache and return ---
    _cache_set(content_hash, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
