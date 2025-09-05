import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import os

DIR_PATH = 'all_jsons/categorized_cases.json'
MIN_DIST = 0.2
N_NEIGHBORS = 10

def parse_case_metadata(metadata_string, count):
    """Parse the JSON metadata from the case value string"""
    try:
        # Extract JSON from the markdown code block
        json_match = re.search(r'```json\n(.*?)\n```', metadata_string, re.DOTALL)
        json_str = json_match.group(1)
        metadata = json.loads(json_str)
        return metadata
    except Exception as e:
        print("COUNT: ", count)
        print(f"Warning: Error processing metadata: {e}")
        return {"ai_material": False, "category": ["Unknown"]}

def load_and_process_data():
    """Load JSON data and process it for embeddings"""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_cases = []
    total = 0
    for casename, metadata_string in data.items():
        total += 1
        metadata = parse_case_metadata(metadata_string, total)
        
        categories = metadata.get('category', ['Unknown'])
        main_category = categories[0] if isinstance(categories, list) and len(categories) > 0 else 'Unknown'
        
        case_info = {
            'title': casename,
            'summary': casename,  # Using casename as summary since no separate summary is provided
            'main_category': main_category,
            'ai_material': metadata.get('ai_material', False),
            'all_categories': categories
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

    # Save TF-IDF embeddings
    np.save(os.path.join(SAVE_DIR, "sent_trans_embeddings.npy"), features)
    
    return features, categories, titles

if __name__ == "__main__":
    try:
        embedding, categories, titles = main()
        print("Embedding generation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")