import os
import time
import openai
import re
from pathlib import Path
from PyPDF2 import PdfReader
from openai.error import RateLimitError, InvalidRequestError

# === CONFIG ===
USE_CLOUD = False
DATA_DIR = "data"
openai.api_key_path = ".env"
MAX_RETRIES = 5

# === Extract text ===
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# === Retry-safe GPT call ===
def gpt_chat(messages, model="gpt-4-1106-preview"):
    for attempt in range(MAX_RETRIES):
        try:
            return openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0
            )
        except RateLimitError as e:
            wait_time = parse_retry_after(str(e)) or (5 * (attempt + 1))
            print(f"Rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        except InvalidRequestError as e:
            print(f"Invalid request: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(3)
    print("Max retries exceeded.")
    return None

def parse_retry_after(error_text):
    """Extract wait time in seconds from error message."""
    match = re.search(r"try again in ([\d.]+)s", error_text)
    return float(match.group(1)) if match else None

# === Materiality Check ===
def is_ai_material(text):
    messages = [{
        "role": "user",
        "content": f"""
You are a legal expert specializing in AI-related litigation.

Is AI involved in this case? AI must be directly tied to the legal claim (IE antitrust, privacy invasions), not just mentioned offhandedly (used to describe someone, but never mentioned again).

Respond with one word: "Yes" or "No".

CASE:
\"\"\"
{text[:8000]}  # limit input to avoid TPM errors
\"\"\"
        """.strip()
    }]
    response = gpt_chat(messages)
    if not response: return False
    return response.choices[0].message['content'].strip().lower() == "yes"

# === Classification Step ===
def classify_legal_category(text):
    messages = [{
        "role": "user",
        "content": f"""
You are a legal expert specializing in AI-related litigation. Classify the case based on the primary legal claim.

Only choose from:
- AI in Legal Proceedings
- Antitrust
- Consumer Protection
- IP Law
- Privacy and Data Protection
- Tort
- Justice and Equity

Return only one category.

CASE:
\"\"\"
{text[:8000]}  # limit input again
\"\"\"
        """.strip()
    }]
    response = gpt_chat(messages)
    if not response: return "Unknown"
    return response.choices[0].message['content'].strip()

# === Full classification pipeline ===
def classify_case(file_path):
    case_text = extract_text_from_pdf(file_path)
    if not case_text:
        return "Empty Document"

    if is_ai_material(case_text):
        return classify_legal_category(case_text)
    else:
        return "Unrelated"

# === Main loop ===
def main():
    input_dir = Path(DATA_DIR)
    files = list(input_dir.glob("*.pdf"))

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        try:
            result = classify_case(file_path)
            print(f" → Classification: {result}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    main()
