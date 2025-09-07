import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# File paths
DIR_PATH = 'court_pdfs_text.json'
SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_and_process_data():
    """Load JSON data (case_name -> full_text) and return dataframe + list"""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_cases = []
    for casename, full_text in data.items():
        case_info = {
            'title': casename,
            'text': full_text if isinstance(full_text, str) else ""
        }
        processed_cases.append(case_info)
    
    df = pd.DataFrame(processed_cases)
    return df, processed_cases

def create_sentence_transformer_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings using a HuggingFace SentenceTransformer model"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model

def main():
    """Main function to run the embedding pipeline"""
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    titles = [case['title'] for case in processed_cases]
    texts = [case['text'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create embeddings with SentenceTransformers
    print("Creating embeddings with SentenceTransformers...")
    features, model = create_sentence_transformer_embeddings(texts)
    print(f"Created embeddings of shape {features.shape}")

    # Save embeddings
    np.save(os.path.join(SAVE_DIR, "sent_trans_embeddings.npy"), features)

    return features, titles

if __name__ == "__main__":
    try:
        embeddings, titles = main()
        print("Embedding generation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")
