"""
Offline DSPy prompt optimization for Challenge 2 (Document Data Extraction).

Uses MIPROv2 to optimize the extraction prompt against training data.
Run this script locally, then copy the optimized prompt into main.py.

Usage:
    export GEMINI_API_KEY=...
    python3 optimize_prompt.py
"""

import json
import os
import re
import sys
import psycopg2
import dspy
from difflib import SequenceMatcher

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    # Try loading from .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    GEMINI_API_KEY = line.strip().split("=", 1)[1]

TRAINING_DB = {
    "host": "35.234.124.49",
    "port": 5432,
    "dbname": "hackathon_training",
    "user": "hackathon_reader",
    "password": "ReadOnly2025hack",
}

# Fields and their comparison types
ENUM_FIELDS = {"state", "assetType", "concludedAs", "contractRegime", "actionOnInsurancePeriodTermination"}
DATE_FIELDS = {"startAt", "endAt", "concludedAt"}
NUMBER_FIELDS = {"installmentNumberPerInsurancePeriod", "insurancePeriodMonths"}
BOOLEAN_FIELDS = set()  # premium.isCollection is nested
STRING_FIELDS = {"contractNumber", "insurerName", "noticePeriod", "regPlate", "latestEndorsementNumber"}
ALL_FIELDS = ENUM_FIELDS | DATE_FIELDS | NUMBER_FIELDS | STRING_FIELDS | {"premium", "note"}


# --- Document classification (same as main.py) ---
def classify_doc(filename, ocr_text):
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


def extract_amendment_number(filename, ocr_text):
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


def prepare_documents_text(documents):
    """Classify, sort, and concatenate documents into a single text block."""
    classified = []
    for doc in documents:
        fn = doc.get("filename", "unknown")
        ocr = doc.get("ocr_text", "")
        dtype = classify_doc(fn, ocr)
        am_num = extract_amendment_number(fn, ocr) if dtype == "amendment" else None
        classified.append({"filename": fn, "ocr_text": ocr, "type": dtype, "am_num": am_num})

    type_order = {"main": 0, "amendment": 1, "other": 2, "vpp": 3}
    classified.sort(key=lambda d: (type_order.get(d["type"], 2), int(d["am_num"]) if d["am_num"] and d["am_num"].isdigit() else 0))

    text = ""
    for doc in classified:
        label = {"main": "MAIN CONTRACT", "amendment": "AMENDMENT", "vpp": "GENERAL TERMS (VPP)", "other": "DOCUMENT"}.get(doc["type"], "DOCUMENT")
        suffix = f" #{doc['am_num']}" if doc["am_num"] else ""
        text += f"### [{label}{suffix}] {doc['filename']}\n"
        ocr = doc["ocr_text"]
        if doc["type"] == "vpp" and len(ocr) > 2000:
            text += ocr[:2000] + "\n[... VPP truncated ...]\n"
        else:
            text += ocr + "\n"
        text += "\n---\n\n"
    return text


# --- Metric ---
def fuzzy_match(a, b, threshold=0.85):
    """Fuzzy string match."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    a_str = str(a).strip().lower()
    b_str = str(b).strip().lower()
    if a_str == b_str:
        return 1.0
    return SequenceMatcher(None, a_str, b_str).ratio()


def compare_field(field, predicted, expected):
    """Compare a single field, return (score, feedback_str or None)."""
    pred_val = predicted.get(field)
    exp_val = expected.get(field)

    if field == "premium":
        # Nested comparison
        pred_p = predicted.get("premium", {})
        exp_p = expected.get("premium", {})
        score = 0.0
        feedback = []
        # currency
        if str(pred_p.get("currency", "")).lower() == str(exp_p.get("currency", "")).lower():
            score += 0.5
        else:
            feedback.append(f"premium.currency: got '{pred_p.get('currency')}', expected '{exp_p.get('currency')}'")
        # isCollection
        if pred_p.get("isCollection") == exp_p.get("isCollection"):
            score += 0.5
        else:
            feedback.append(f"premium.isCollection: got {pred_p.get('isCollection')}, expected {exp_p.get('isCollection')}")
        return score, "; ".join(feedback) if feedback else None

    if field == "note":
        # Fuzzy match for note — lenient
        ratio = fuzzy_match(pred_val, exp_val, 0.3)
        if ratio < 0.3:
            return 0.0, f"note: low similarity ({ratio:.2f})"
        return min(ratio / 0.7, 1.0), None  # scale up

    if field in ENUM_FIELDS:
        if pred_val == exp_val:
            return 1.0, None
        return 0.0, f"{field}: got '{pred_val}', expected '{exp_val}'"

    if field in DATE_FIELDS:
        if pred_val == exp_val:
            return 1.0, None
        if pred_val is None and exp_val is None:
            return 1.0, None
        return 0.0, f"{field}: got '{pred_val}', expected '{exp_val}'"

    if field in NUMBER_FIELDS:
        if pred_val == exp_val:
            return 1.0, None
        try:
            if abs(float(pred_val) - float(exp_val)) / max(float(exp_val), 1) <= 0.1:
                return 0.8, f"{field}: close but not exact ({pred_val} vs {exp_val})"
        except (TypeError, ValueError):
            pass
        return 0.0, f"{field}: got {pred_val}, expected {exp_val}"

    if field in STRING_FIELDS:
        if pred_val is None and exp_val is None:
            return 1.0, None
        ratio = fuzzy_match(pred_val, exp_val)
        if ratio >= 0.85:
            return 1.0, None
        if ratio >= 0.6:
            return 0.5, f"{field}: partial match ({ratio:.2f}): '{pred_val}' vs '{exp_val}'"
        return 0.0, f"{field}: got '{pred_val}', expected '{exp_val}'"

    return 0.0, f"{field}: unknown field type"


def extraction_metric(example, prediction, trace=None):
    """Score extraction result against expected output. Returns 0.0-1.0."""
    try:
        predicted = json.loads(prediction.result_json)
    except (json.JSONDecodeError, AttributeError):
        return 0.0

    expected = json.loads(example.result_json) if isinstance(example.result_json, str) else example.result_json

    total_score = 0.0
    total_fields = 0
    feedback_items = []

    for field in ALL_FIELDS:
        score, feedback = compare_field(field, predicted, expected)
        total_score += score
        total_fields += 1
        if feedback:
            feedback_items.append(feedback)

    final_score = total_score / total_fields if total_fields > 0 else 0.0

    if trace is not None and feedback_items:
        # Provide textual feedback for optimizers that use it
        dspy.suggest(final_score > 0.8, f"Issues: {'; '.join(feedback_items[:5])}")

    return final_score


# --- Load training data ---
def load_training_data():
    """Load training examples from the training DB."""
    conn = psycopg2.connect(**TRAINING_DB)
    cur = conn.cursor()
    cur.execute("SELECT input, expected_output FROM training_data WHERE challenge_id = 2")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    examples = []
    for inp, expected in rows:
        documents = inp["documents"]
        docs_text = prepare_documents_text(documents)
        expected_json = json.dumps(expected, ensure_ascii=False)
        examples.append(dspy.Example(
            documents_text=docs_text,
            result_json=expected_json,
        ).with_inputs("documents_text"))

    return examples


def generate_synthetic_examples(real_examples, num_synthetic=8):
    """Use Gemini to generate synthetic training variations."""
    print(f"Generating {num_synthetic} synthetic examples from {len(real_examples)} real ones...")

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    synthetic = []
    per_real = num_synthetic // len(real_examples)

    for i, example in enumerate(real_examples):
        expected = json.loads(example.result_json)
        prompt = f"""You are generating synthetic training data for an insurance document extraction system.

Given this real expected output from a Czech insurance document:
{json.dumps(expected, ensure_ascii=False, indent=2)}

Generate {per_real} DIFFERENT variations of this expected output. For each variation:
- Change the contract number format slightly (different digits)
- Change dates (different years/months, keep DD.MM.YYYY format)
- Sometimes change the insurer name to another Czech insurance company (e.g., ČSOB Pojišťovna, Kooperativa, Generali, Uniqa, AXA)
- Sometimes change premium currency to "eur"
- Vary the note text (keep Czech insurance context)
- Keep the same field structure and enum values
- Make realistic but distinct variations

Also for each variation, generate a plausible short OCR text snippet (500-1000 chars) that would contain the key information for that variation, written as if it were a Czech insurance contract. Include the contract number, dates, insurer name, and key terms.

Return a JSON array where each element has:
- "expected": the expected output object
- "ocr_snippet": a plausible OCR text snippet

Return ONLY valid JSON, no markdown."""

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )
            variations = json.loads(response.text.strip())
            if not isinstance(variations, list):
                variations = [variations]

            for var in variations[:per_real]:
                ocr_text = var.get("ocr_snippet", "")
                expected_out = var.get("expected", {})
                # Build a minimal documents_text
                docs_text = f"### [MAIN CONTRACT] synthetic_{i}.pdf\n{ocr_text}\n\n---\n\n"
                synthetic.append(dspy.Example(
                    documents_text=docs_text,
                    result_json=json.dumps(expected_out, ensure_ascii=False),
                ).with_inputs("documents_text"))
                print(f"  Generated synthetic example {len(synthetic)}")

        except Exception as e:
            print(f"  Warning: Failed to generate synthetic data from example {i}: {e}")

    return synthetic


# --- DSPy Signature and Module ---
class ExtractInsuranceFields(dspy.Signature):
    """You are an expert Czech insurance document analyst. Extract structured CRM fields from the provided insurance contract documents.

IMPORTANT RULES:
- Documents are provided in order: main contract first, then amendments (dodatky) in chronological order, then general terms (VPP).
- Amendments OVERRIDE values from the base contract. Always use the LATEST value for each field.
- Return null for fields you cannot find in the documents.
- Dates MUST be in DD.MM.YYYY format, zero-padded.
- The note field should be a concise Czech summary of special conditions, exclusions, and notable coverage details.

FIELDS TO EXTRACT (return as JSON):
contractNumber, insurerName, state (draft|accepted|cancelled), assetType (other|vehicle),
concludedAs (agent|broker), contractRegime (individual|frame|fleet|coinsurance),
startAt (DD.MM.YYYY), endAt (DD.MM.YYYY or null), concludedAt (DD.MM.YYYY),
installmentNumberPerInsurancePeriod (1|2|4|12), insurancePeriodMonths (1|3|6|12),
premium.currency (czk|eur), premium.isCollection (bool),
actionOnInsurancePeriodTermination (auto-renewal|policy-termination),
noticePeriod (six-weeks|three-months|two-months|one-month|null),
regPlate (string|null), latestEndorsementNumber (string|null), note (string|null).

If RENOMIA or makléř is mentioned, concludedAs=broker. If auto-renews, endAt=null.
Return ONLY a valid JSON object."""

    documents_text: str = dspy.InputField(desc="Concatenated OCR text from classified and sorted insurance documents")
    result_json: str = dspy.OutputField(desc="JSON object with the 17+ CRM fields extracted from the documents")


class InsuranceExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractInsuranceFields)

    def forward(self, documents_text):
        return self.extract(documents_text=documents_text)


# --- Main optimization ---
def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    # Configure DSPy with Gemini via OpenAI-compatible endpoint
    lm = dspy.LM(
        "openai/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.1,
        max_tokens=8192,
    )
    dspy.configure(lm=lm)

    # Load real training data
    print("Loading training data...")
    real_examples = load_training_data()
    print(f"Loaded {len(real_examples)} real training examples")

    # Generate synthetic examples
    synthetic_examples = generate_synthetic_examples(real_examples, num_synthetic=10)
    print(f"Generated {len(synthetic_examples)} synthetic examples")

    # Combine: real + synthetic for training, real for validation
    trainset = real_examples + synthetic_examples
    valset = real_examples  # Always validate against real data

    print(f"\nTraining set: {len(trainset)} examples")
    print(f"Validation set: {len(valset)} examples")

    # Create the module
    student = InsuranceExtractor()

    # First, evaluate baseline
    print("\n--- Baseline Evaluation ---")
    evaluate = dspy.Evaluate(
        devset=valset,
        metric=extraction_metric,
        num_threads=1,
        display_progress=True,
    )
    baseline_score = evaluate(student)
    print(f"Baseline score: {baseline_score:.4f}")

    # Run MIPROv2 optimization
    print("\n--- Running MIPROv2 Optimization ---")
    optimizer = dspy.MIPROv2(
        metric=extraction_metric,
        auto="light",
        num_threads=1,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        verbose=True,
    )

    optimized = optimizer.compile(
        student,
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,
    )

    # Evaluate optimized version
    print("\n--- Optimized Evaluation ---")
    optimized_score = evaluate(optimized)
    print(f"Optimized score: {optimized_score:.4f}")
    print(f"Improvement: {optimized_score - baseline_score:+.4f}")

    # Extract the optimized prompt
    print("\n--- Extracting Optimized Prompt ---")
    # The optimized instructions are stored in the predict module's signature
    try:
        sig = optimized.extract.signature
        instructions = sig.instructions
        print(f"\nOptimized instructions:\n{'='*60}")
        print(instructions)
        print('='*60)
    except AttributeError:
        print("Could not extract instructions directly, inspecting module...")
        print(optimized.extract)

    # Save the optimized model
    output_path = os.path.join(os.path.dirname(__file__), "optimized_extractor.json")
    optimized.save(output_path)
    print(f"\nOptimized model saved to: {output_path}")

    # Also save just the prompt for easy copy-paste
    prompt_path = os.path.join(os.path.dirname(__file__), "optimized_prompt.txt")
    try:
        with open(prompt_path, "w") as f:
            f.write(instructions)
        print(f"Optimized prompt saved to: {prompt_path}")
    except Exception:
        pass

    # Detailed comparison on real examples
    print("\n--- Detailed Comparison on Real Examples ---")
    for i, example in enumerate(real_examples):
        print(f"\nExample {i+1}:")
        expected = json.loads(example.result_json)

        # Run optimized
        pred = optimized(documents_text=example.documents_text)
        try:
            predicted = json.loads(pred.result_json)
        except (json.JSONDecodeError, AttributeError):
            print(f"  ERROR: Could not parse prediction")
            continue

        correct = 0
        total = 0
        for field in ALL_FIELDS:
            score, feedback = compare_field(field, predicted, expected)
            total += 1
            if score >= 0.85:
                correct += 1
            else:
                print(f"  WRONG: {feedback}")
        print(f"  Score: {correct}/{total}")

    if optimized_score >= baseline_score:
        print(f"\n>>> Optimization successful! Score improved from {baseline_score:.4f} to {optimized_score:.4f}")
        print(f">>> Copy the prompt from {prompt_path} into main.py EXTRACTION_PROMPT")
    else:
        print(f"\n>>> Optimization did not improve. Baseline: {baseline_score:.4f}, Optimized: {optimized_score:.4f}")
        print(">>> Keeping original prompt.")


if __name__ == "__main__":
    main()
