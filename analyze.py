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

# === Extract and clean text ===
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return clean_legal_text(raw_text)

def clean_legal_text(text, max_chars=50000):
    """Clean and filter legal document text to remove noise and irrelevant content."""
    if not text:
        return ""
    
    # Limit length - take first half only
    text = text[:max_chars]
    
    # Remove common PDF artifacts and noise
    import string
    
    # Split into lines for line-by-line filtering
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove lines that are mostly numbers (page numbers, case numbers, etc.)
        if len(line) > 0 and sum(c.isdigit() for c in line) / len(line) > 0.7:
            continue
            
        # Remove lines that are mostly punctuation or special characters
        if len(line) > 0 and sum(c in string.punctuation for c in line) / len(line) > 0.5:
            continue
            
        # Remove very short lines (likely artifacts)
        if len(line) < 3:
            continue
            
        # Remove lines that look like headers/footers (all caps, short)
        if len(line) < 50 and line.isupper():
            continue
            
        # Remove lines with excessive spacing or weird characters
        if '  ' in line and len(line.replace(' ', '')) < len(line) * 0.3:
            continue
            
        # Clean the line itself
        cleaned_line = clean_line(line)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    # Rejoin and do final cleanup
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove excessive whitespace
    import re
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Max 2 newlines
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Normalize spaces
    
    return cleaned_text.strip()

def clean_line(line):
    """Clean individual line of text."""
    import re
    
    # Remove non-printable characters except newlines and tabs
    line = ''.join(c for c in line if c.isprintable() or c in '\n\t')
    
    # Fix common OCR errors and artifacts
    line = re.sub(r'[^\w\s\.,;:!?()"\'-]', ' ', line)  # Keep only common punctuation
    
    # Remove excessive repeated characters
    line = re.sub(r'(.)\1{4,}', r'\1\1\1', line)  # Max 3 repeated chars
    
    # Remove standalone numbers and weird fragments
    line = re.sub(r'\b\d+\b(?!\s*[A-Za-z])', ' ', line)  # Remove standalone numbers
    
    # Clean up spacing
    line = re.sub(r'\s+', ' ', line).strip()
    
    # Skip lines that became too short after cleaning
    if len(line) < 5:
        return ""
        
    return line

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
    # Use cleaned text directly, limited to 15000 chars for API efficiency
    truncated_text = text[:15000]
    
    messages = [{
        "role": "system",
        "content": """You are a legal expert specializing in AI-related litigation. 

Classify cases based on whether AI/algorithms/automated systems are involved in the legal claims.

Categories with specific guidance:

AI in Legal Proceedings: 
- AI systems used IN court processes, legal case management, or litigation tools
- AI affecting judicial decisions or legal outcomes
- Legal technology platforms, e-discovery tools, legal AI assistants
- Look for: litigation software, legal tech, court systems, judicial AI

Antitrust: 
- Market competition, monopolization involving ANY tech companies
- Anti-competitive practices by major platforms or AI companies  
- Look for: market dominance, competition, monopoly, anti-competitive

Consumer Protection: 
- Deceptive practices, unfair business practices with tech/automated systems
- Misleading marketing of tech products or AI capabilities

IP Law: 
- Patents, copyrights, trademarks for AI models or tech
- Training data disputes, AI-generated content ownership

Privacy and Data Protection: 
- Data breaches, unauthorized data collection by automated systems
- Privacy violations involving algorithms or data processing

Tort: 
- Physical harm, emotional distress, negligence involving ANY automated systems
- Defamation, personal injury from tech systems or algorithms
- Look for: negligence, injury, harm, emotional distress, defamation

Justice and Equity: 
- Discrimination or bias by automated systems (hiring, lending, search)
- Civil rights violations involving algorithms or platforms
- Unfair treatment based on algorithmic decisions

CRITICAL FIXES:
1. For antitrust cases involving tech companies use "Antitrust" not Unrelated
2. For tort claims involving any tech systems use "Tort" not Unrelated  
3. For legal technology or court AI systems use "AI in Legal Proceedings"
4. Be more inclusive - tech companies typically use automated systems

Respond with JSON: {"ai_material": true/false, "category": "category_name"}
If truly no automated systems involved, use: {"ai_material": false, "category": "Unrelated"}"""
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
    
    # DEBUG: Print text length and first few words

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