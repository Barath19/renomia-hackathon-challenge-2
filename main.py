"""
Challenge 2: Vyčítání dat ze souborů (Document Data Extraction)

Input:  OCR text from insurance contract documents (main contract + amendments)
Output: Structured CRM fields extracted from the documents
"""

import os
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

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
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


@app.post("/solve")
def solve(payload: dict):
    """
    Extract structured CRM fields from insurance contract documents.

    Input example:
    {
        "documents": [
            {
                "filename": "smlouva_hlavni.pdf",
                "ocr_text": "... OCR extracted text of main contract ..."
            },
            {
                "filename": "dodatek_1.pdf",
                "ocr_text": "... OCR text of amendment 1 ..."
            },
            {
                "filename": "dodatek_2.pdf",
                "ocr_text": "... OCR text of amendment 2 ..."
            }
        ]
    }

    Expected output (all fields from CRM template):
    {
        "contractNumber": "POJ-2024-12345",
        "insurerName": "Generali Česká pojišťovna a.s.",
        "state": "accepted",              // enum: draft | accepted | cancelled
        "assetType": "other",              // enum: other | vehicle
        "concludedAs": "broker",           // enum: agent | broker
        "contractRegime": "individual",    // enum: individual | frame | fleet | coinsurance
        "startAt": "01.01.2024",           // DD.MM.YYYY
        "endAt": null,                     // DD.MM.YYYY or null (doba neurčitá)
        "concludedAt": "15.12.2023",       // DD.MM.YYYY
        "installmentNumberPerInsurancePeriod": 4,  // 1=yearly, 2=semi, 4=quarterly, 12=monthly
        "insurancePeriodMonths": 12,       // 12=yearly, 6=semi, 3=quarterly, 1=monthly
        "premium": {
            "currency": "czk",             // ISO 4217 lowercase
            "isCollection": false          // true if broker collects premium
        },
        "actionOnInsurancePeriodTermination": "auto-renewal",  // auto-renewal | policy-termination
        "noticePeriod": "six-weeks",       // enum or null
        "regPlate": null,                  // only for vehicle insurance
        "latestEndorsementNumber": "3",    // string, highest amendment number or null
        "note": null                       // special conditions summary or null
    }
    """
    # TODO: Implement your solution here
    #
    # Suggested approach:
    # 1. Concatenate OCR text from all documents (main contract + amendments)
    # 2. Send to Gemini with a structured extraction prompt
    # 3. Parse the response into the expected field format
    # 4. For amendments: use the latest values (amendments override base contract)
    # 5. For latestEndorsementNumber: find the highest amendment number

    documents = payload.get("documents", [])

    result = {
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
        "premium": {
            "currency": "czk",
            "isCollection": False,
        },
        "actionOnInsurancePeriodTermination": "auto-renewal",
        "noticePeriod": None,
        "regPlate": None,
        "latestEndorsementNumber": None,
        "note": None,
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
