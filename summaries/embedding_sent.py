import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

DIR_PATH = 'categorized_cases.json'
MIN_DIST = 0.2
N_NEIGHBORS = 10

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

def create_sentence_transformer_embeddings(summaries, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings using a HuggingFace SentenceTransformer model"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(summaries, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model

SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    """Main function to run the embedding pipeline"""
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]
    categories = [case['main_category'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create embeddings with SentenceTransformers
    print("Creating embeddings with SentenceTransformers...")
    features, model = create_sentence_transformer_embeddings(summaries)
    print(f"Created embeddings of shape {features.shape}")

    # Save embeddings
    np.save(os.path.join(SAVE_DIR, "sent_trans_embeddings.npy"), features)
    
    return features, categories, titles

if __name__ == "__main__":
    try:
        embedding, categories, titles = main()
        print("Embedding generation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")
