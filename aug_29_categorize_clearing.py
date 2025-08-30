import json

DATA_DIR="all_jsons/categorized_cases.json"

def main():
    all_court_pdfs = {}
    with open(DATA_DIR) as f:
        all_court_pdfs = json.load(f)
    total = 0
    for title, text in all_court_pdfs.items():
        # Remove the leading and trailing Markdown code block markers
        stripped_text = text.strip().replace("```json\n", "").replace("```", "")
        total += 1
        try:
            # Now, `stripped_text` is a clean JSON string, ready to be loaded
            data = json.loads(stripped_text)
        except json.JSONDecodeError as e:
            print(total)
            print(f"Error decoding JSON for case '{title}': {e}")
            print(f"Problematic string: {stripped_text}")

if __name__ == "__main__":
    main()