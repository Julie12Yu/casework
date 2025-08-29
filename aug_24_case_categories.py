import json
from openai import OpenAI
import sys

def get_raw_response(prompt, model="gpt-4.1", **kwargs):
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return prompt, response.choices[0].message.content


def classify_text(title, text):
    print(f"Classifying: {title}")
    prompt = "You are a legal expert specialized in categorizing cases based on the aspect of the case specifically related to AI." + \
f"The case title is: {title}" + \
f"The summary of the case is: {text}" + \
'''Reading a summary of the case, then classify the case based on the definition below:

1. Antitrust: refers to cases where the defendant is accused of market competition, monopolization involving ANY tech companies, or anti-competitive practices by major platforms or AI companies.
2. IP Law: refers to cases where the defendant is accused of patents, copyrights, trademarks for AI models or tech, or training data disputes, AI-generated content ownership.
3. Privacy and Data Protection: refers to cases where the defendant is accused of data breaches, unauthorized data collection by automated systems, or privacy violations involving algorithms or data processing.
4. Tort: refers to cases where the defendant is accused of physical harm, emotional distress, negligence involving ANY automated systems, or defamation, personal injury from tech systems or algorithms.
5. Justice and Equity: refers to cases where the defendant is accused of discrimination or bias **caused by AI, automated systems, or algorithms** (e.g., hiring, lending, search). Do not use this category for discrimination cases without automation.
6. Consumer Protection: refers to cases where the defendant is accused of deceptive practices, unfair business practices with tech/automated systems, or misleading marketing of tech products or AI capabilities.
7. AI in Legal Proceedings: refers to cases where AI systems are merely used in the court processes, legal case management, or litigation tools. The core contention is not about AI, but AI tools have been used in the litigation process.
8. Unrelated: refers to cases that have no meaningful connection to artificial intelligence (AI), machine learning (ML), or automated systems. If the case involves discrimination, privacy, or other issues **without automation/AI/algorithmic involvement**, classify as Unrelated.

Rule 1: Classify the cases from the categories above on the aspect of the case specifically related to AI(i.e., AI in Legal Proceedings, Antitrust, Consumer Protection, IP Law, Tort, Justice and Equity, Unrelated)
Rule 2: If multiple categories apply, use all relevant categories. If no category applies, use "Unrelated".
Rule 3: Respond with JSON: {"ai_material": true/false, "category": ["category_name1", "category_name2", ...]}.'''

    _, output = get_raw_response(prompt, model="gpt-4o-mini")
    return output

def summarize_text(title, text):
    print(f"Summarizing: {title}")
    prompt = (
        f"You are a legal expert. Please summarize the case: {title}. \n\n" +
        f"Here is the context of the case: {text}. \n\n" +
        "Rule 1: Summarize the case in a few sentences. \n\n" +
        "Rule 2: If the case is related to artificial intelligence (AI) or machine learning (ML) or automated systems, please focus on how the AI is involved in the summary. \n\n" +
        '''Rule 3: Respond with JSON: {"summary": "summary_text"}.'''
    )

    _, output = get_raw_response(prompt, model="gpt-4o-mini")
    return output
    

def parse_output(output):
    try:
        return json.loads(output)
    except:
        raise ValueError("Invalid output")
    

def main(DATA_DIR):
    all_court_pdfs = {}
    with open(DATA_DIR) as f:
        all_court_pdfs = json.load(f)

    case_to_categories = {}
    for title, text in all_court_pdfs.items():
        summary = summarize_text(title, text)
        categories = classify_text(title, summary)
        case_to_categories.setdefault(title, categories)

    # You can optionally save the result to a file or print it
    with open("categorized_cases.json", "w", encoding="utf-8") as f:
        json.dump(case_to_categories, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/data.json")
        sys.exit(1)
    data_path = sys.argv[1]
    main(data_path)