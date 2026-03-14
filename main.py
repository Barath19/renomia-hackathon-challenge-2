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


EXTRACTION_PROMPT = """You are an expert Czech insurance document analyst. Extract structured CRM fields from the provided insurance contract documents.

IMPORTANT RULES:
- Documents are provided in order: main contract first, then amendments (dodatky) in chronological order, then general terms (VPP).
- Amendments OVERRIDE values from the base contract. Always use the LATEST value for each field.
- Return null for fields you cannot find in the documents.
- Dates MUST be in DD.MM.YYYY format, zero-padded.
- The note field should be a concise Czech summary of special conditions, exclusions, and notable coverage details. Write it in Czech language.

FIELDS TO EXTRACT:

1. contractNumber (string): The contract number (číslo smlouvy/pojistné smlouvy). Keep the exact format from the document including spaces.
2. insurerName (string): The insurer/pojistitel name. Use the official name.
3. state (enum): "draft" | "accepted" | "cancelled" — Default "accepted" unless explicitly stated otherwise.
4. assetType (enum): "other" | "vehicle" — Use "vehicle" only for vehicle/auto insurance (pojištění vozidel, autopojištění). Otherwise "other".
5. concludedAs (enum): "agent" | "broker" — If RENOMIA or any makléř/zprostředkovatel is mentioned, use "broker". Default "broker".
6. contractRegime (enum): "individual" | "frame" | "fleet" | "coinsurance" — "frame" for rámcová smlouva, "fleet" for flotilové, "coinsurance" for spolupojištění. Default "individual".
7. startAt (string, DD.MM.YYYY): Insurance start date (počátek pojištění).
8. endAt (string or null): Insurance end date (konec pojištění). null if indefinite term (doba neurčitá). IMPORTANT: If the contract auto-renews (actionOnInsurancePeriodTermination is "auto-renewal"), set endAt to null — even if there was an initial fixed term, after auto-renewal it becomes indefinite.
9. concludedAt (string, DD.MM.YYYY): Date the contract was concluded/signed (datum uzavření/sjednání). Often same as startAt.
10. installmentNumberPerInsurancePeriod (number): Payment frequency per insurance period. "ročně"=1, "pololetně"=2, "čtvrtletně"=4, "měsíčně"=12. Check amendments for changes.
11. insurancePeriodMonths (number): Insurance period length in months. "roční/12 měsíců"=12, "pololetní/6 měsíců"=6, "čtvrtletní/3 měsíce"=3, "měsíční/1 měsíc"=1. Check amendments for changes.
12. premium.currency (string): Currency in ISO 4217 lowercase. Usually "czk". Could be "eur".
13. premium.isCollection (boolean): true if premiums are collected through the broker/makléř (inkaso makléře/zprostředkovatele). false if paid directly to insurer.
14. actionOnInsurancePeriodTermination (enum): "auto-renewal" | "policy-termination" — "auto-renewal" if contract auto-renews (automatické prodloužení/prolongace). "policy-termination" if it ends after fixed term without renewal.
15. noticePeriod (string or null): Notice period. Use EXACTLY one of: "six-weeks", "three-months", "two-months", "one-month", or null. null if fixed-term with no notice period.
16. regPlate (string or null): Vehicle registration plate (SPZ/RZ). null for non-vehicle insurance.
17. latestEndorsementNumber (string or null): The identifier of the latest amendment/endorsement (dodatek). Could be a number like "3" or alphanumeric like "DOP 098". null if no amendments.
18. note (string or null): Concise Czech summary (1-3 sentences) of key special conditions, exclusions, territorial scope, and notable coverage details. Write in Czech. null if nothing notable.

RESPOND WITH ONLY A VALID JSON OBJECT matching this exact structure (no markdown, no backticks):
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
    if "isCollection" not in result["premium"]:
        result["premium"]["isCollection"] = False

    # Endorsement number — prefer Gemini's alphanumeric, fallback to deterministic
    if det_endorsement:
        gemini_num = result.get("latestEndorsementNumber")
        if not gemini_num:
            result["latestEndorsementNumber"] = det_endorsement

    return result


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

    # Sort: main first, amendments by number, then other, VPP last
    type_order = {"main": 0, "amendment": 1, "other": 2, "vpp": 3}
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

    # --- Cache and return ---
    _cache_set(content_hash, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
