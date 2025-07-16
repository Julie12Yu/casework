import os
import time
import openai
import re
from pathlib import Path
from PyPDF2 import PdfReader
from openai.error import RateLimitError, InvalidRequestError
import json

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
def gpt_chat(messages, model="gpt-4-turbo-preview"):  # Use latest model
    for attempt in range(MAX_RETRIES):
        try:
            return openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=50,  # Limit tokens for classification
                timeout=30  # Add timeout
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

# === Combined Classification (More Efficient) ===
def classify_case_combined(text):
    # Truncate more intelligently - focus on key sections
    truncated_text = get_relevant_sections(text)
    
    messages = [{
        "role": "system",
        "content": """You are a legal expert specializing in AI-related litigation. 

STEP 1: Determine if AI/algorithms/machine learning are materially involved in the legal claims.

Look for these AI-related keywords and concepts:
- Artificial intelligence, AI, machine learning, algorithms, automated systems
- Neural networks, deep learning, natural language processing
- Recommendation systems, search algorithms, ranking systems
- Automated decision-making, algorithmic bias, AI discrimination
- Chatbots, virtual assistants, AI-powered tools
- Computer vision, image recognition, facial recognition
- Predictive analytics, AI models, training data

AI is MATERIAL if these technologies are:
- Central to the business being sued
- Part of the alleged harmful conduct
- Mentioned in the legal claims or causes of action
- Used in decision-making that's being challenged

STEP 2: If AI is material, classify by the PRIMARY legal harm:

**AI in Legal Proceedings**: AI/algorithms used in courts, legal processes, case management, or litigation tools that affect legal outcomes

**Justice and Equity**: 
- Discrimination by AI hiring, lending, housing tools
- Algorithmic bias in services, recommendations, search results
- Civil rights violations by automated systems
- Platform bias against protected groups

**Antitrust**: Anti-competitive practices involving AI companies or AI technology market dominance

**Consumer Protection**: Deceptive AI marketing, unfair AI business practices, misleading AI product claims

**IP Law**: Patents, copyrights, trademarks for AI models, training data disputes, AI-generated content ownership

**Privacy and Data Protection**: Unauthorized data collection/use by AI, AI-related data breaches, privacy violations

**Tort**: Physical/emotional harm caused by AI systems, negligence in AI deployment, defamation by AI

KEY INSIGHT: If you see discrimination, bias, or unfair treatment involving algorithms/AI → "Justice and Equity"
If you see AI in court/legal systems → "AI in Legal Proceedings"

Respond with JSON: {"ai_material": true/false, "category": "category_name"}
If not AI-related, use: {"ai_material": false, "category": "Unrelated"}"""
    }, {
        "role": "user", 
        "content": f"Analyze this case:\n\n{truncated_text}"
    }]
    
    response = gpt_chat(messages, model="gpt-4-turbo-preview")
    if not response:
        return "Unknown"
    
    try:
        result = json.loads(response.choices[0].message['content'])
        return result['category'] if result['ai_material'] else "Unrelated"
    except:
        # Fallback to text parsing
        content = response.choices[0].message['content'].strip()
        if "Unrelated" in content:
            return "Unrelated"
        # Extract category from response
        for category in ["AI in Legal Proceedings", "Antitrust", "Consumer Protection", 
                        "IP Law", "Privacy and Data Protection", "Tort", "Justice and Equity"]:
            if category in content:
                return category
        return "Unknown"

def get_relevant_sections(text, max_chars=10000):
    """Extract most relevant sections for classification."""
    # Look for key legal sections
    sections = []
    
    # Split by common legal section markers
    patterns = [
        r'BACKGROUND',
        r'FACTUAL ALLEGATIONS',
        r'STATEMENT OF FACTS',
        r'CLAIMS FOR RELIEF',
        r'CAUSE OF ACTION',
        r'COMPLAINT',
        r'SUMMARY',
        r'NATURE OF THE ACTION'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start = match.start()
            end = min(start + 2000, len(text))  # Take 2000 chars after each section
            sections.append(text[start:end])
    
    # If no structured sections found, take beginning and middle
    if not sections:
        sections = [text[:5000], text[len(text)//2:len(text)//2+5000]]
    
    combined = "\n\n".join(sections)
    return combined[:max_chars]

# === Enhanced Two-Step Approach (Alternative) ===
def is_ai_material_enhanced(text):
    """More precise AI materiality check with examples."""
    messages = [{
        "role": "system",
        "content": """You are a legal expert. Determine if AI is MATERIALLY involved in the legal claims.

AI is MATERIAL if:
- AI systems directly caused or contributed to the alleged harm
- AI technology is the subject of the legal dispute
- AI processes are central to the legal violation
- The case involves algorithmic bias, discrimination, or unfair AI decision-making
- AI systems are part of the business model being challenged
- The lawsuit involves AI hiring tools, recommendation systems, or automated decision-making

AI is NOT MATERIAL if:
- Only mentioned in passing or background
- Used as a comparison or metaphor
- Company happens to use AI but claims don't specifically involve AI technology

EXAMPLES:
- MATERIAL: "AI hiring tool discriminated against candidates" 
- MATERIAL: "Algorithm biased search results against plaintiff"
- MATERIAL: "AI model trained on copyrighted content"
- NOT MATERIAL: "Company CEO mentioned AI in earnings call"
- NOT MATERIAL: "AI mentioned as future technology trend"

Be more inclusive - if AI is mentioned in context of the business operations or decision-making that's being challenged, it's likely material.

Respond ONLY: "MATERIAL" or "NOT_MATERIAL\""""
    }, {
        "role": "user",
        "content": f"Case text:\n\n{text[:8000]}"
    }]
    
    response = gpt_chat(messages, model="gpt-4-turbo-preview")
    if not response:
        return False
    return "MATERIAL" in response.choices[0].message['content'].upper()

def classify_legal_category_enhanced(text):
    """Enhanced classification with better category definitions."""
    messages = [{
        "role": "system", 
        "content": """Classify this AI-related legal case by PRIMARY legal theory:

**AI in Legal Proceedings**: AI systems used in courts, legal decision-making, case management, litigation support affecting legal outcomes

**Antitrust**: Market competition, monopolization, anti-competitive practices involving AI companies or technology

**Consumer Protection**: Deceptive AI marketing, unfair AI business practices, consumer fraud with AI products/services

**IP Law**: Patents, copyrights, trademarks, trade secrets for AI models, training data, AI-generated content

**Privacy and Data Protection**: Data breaches, privacy violations, unauthorized data collection/use by AI systems

**Tort**: Personal injury, negligence, defamation, emotional distress caused by AI systems or decisions

**Justice and Equity**: Discrimination, bias, civil rights violations, algorithmic fairness issues, AI hiring/lending/housing discrimination

IMPORTANT GUIDANCE:
- For cases involving AI discrimination in hiring, lending, housing, or services → "Justice and Equity"
- For cases about AI bias in decision-making systems → "Justice and Equity" 
- For platform/search algorithm bias claims → "Justice and Equity"
- For AI systems used in legal/judicial processes → "AI in Legal Proceedings"
- Don't default to "Unrelated" - if AI is mentioned in context of business operations, it's likely material

Look at the PRIMARY legal claims and allegations, not just secondary issues.
Respond with ONLY the category name."""
    }, {
        "role": "user",
        "content": f"Case details:\n\n{text[:10000]}"
    }]
    
    response = gpt_chat(messages, model="gpt-4-turbo-preview")
    if not response:
        return "Unknown"
    
    content = response.choices[0].message['content'].strip()
    
    # Validate response is one of our categories
    valid_categories = [
        "AI in Legal Proceedings", "Antitrust", "Consumer Protection", 
        "IP Law", "Privacy and Data Protection", "Tort", "Justice and Equity"
    ]
    
    for category in valid_categories:
        if category in content:
            return category
    
    return "Unknown"

# === Full classification pipeline ===
def classify_case(file_path, use_combined=True):
    case_text = extract_text_from_pdf(file_path)
    if not case_text:
        return "Empty Document"

    if use_combined:
        result = classify_case_combined(case_text)
        print(f"   API Response: {result}")
        return result
    else:
        # Two-step approach
        if is_ai_material_enhanced(case_text):
            return classify_legal_category_enhanced(case_text)
        else:
            return "Unrelated"

# === Main loop ===
def main():
    input_dir = Path(DATA_DIR)
    files = list(input_dir.glob("*.pdf"))

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        try:
            result = classify_case(file_path, use_combined=True)
            print(f" → Classification: {result}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    main()