import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os

DIR_PATH = 'categorized_cases.json'
SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Legal-BERT model & tokenizer
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_and_process_data():
    """Load JSON data and process it for embeddings"""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_cases = []
    for casename, case_data in data.items():
        # Extract fields from new format
        summary = case_data.get("summary", casename)  # fallback to casename
        categories_obj = case_data.get("categories", {})
        categories = categories_obj.get("category", ["Unknown"])
        ai_material = categories_obj.get("ai_material", False)
        
        main_category = categories[0] if isinstance(categories, list) and len(categories) > 0 else "Unknown"
        
        case_info = {
            "title": casename,
            "summary": summary,
            "main_category": main_category,
            "ai_material": ai_material,
            "all_categories": categories,
        }
        processed_cases.append(case_info)
    
    df = pd.DataFrame(processed_cases)
    return df, processed_cases

def create_legalbert_embeddings(texts, batch_size=16, device=None):
    """Create embeddings using Legal-BERT"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)

def main():
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]
    categories = [case['main_category'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create Legal-BERT embeddings
    print("Creating Legal-BERT embeddings...")
    features = create_legalbert_embeddings(summaries)
    print(f"Created embeddings of shape {features.shape}")
    
    # Save embeddings
    np.save(os.path.join(SAVE_DIR, "legalbert_embeddings.npy"), features)
    
    return features, categories, titles

if __name__ == "__main__":
    try:
        embedding, categories, titles = main()
        print("Legal-BERT embedding pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")
