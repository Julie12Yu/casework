import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

DIR_PATH = 'categorized_cases.json'
MIN_DIST = 0.2
N_NEIGHBORS = 10

def load_and_process_data():
    """Load JSON data and process it for UMAP visualization"""
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

def create_text_features(summaries):
    """Create TF-IDF features from case summaries"""
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=None,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.9
    )
    
    try:
        tfidf_features = vectorizer.fit_transform(summaries)
        return tfidf_features.toarray(), vectorizer
    except ValueError as e:
        print(f"Error creating TF-IDF features: {e}")
        # Fallback: character n-grams
        from sklearn.feature_extraction.text import CountVectorizer
        char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=200)
        char_features = char_vectorizer.fit_transform(summaries)
        return char_features.toarray(), char_vectorizer

SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    """Main function to run the complete interactive UMAP visualization pipeline"""
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]
    categories = [case['main_category'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create text features
    print("Creating TF-IDF features from summaries...")
    features, vectorizer = create_text_features(summaries)
    print(f"Created {features.shape[1]} features")
    
    # Save TF-IDF embeddings
    np.save(os.path.join(SAVE_DIR, "tfidf_embeddings.npy"), features)
    print(f"TF-IDF embeddings saved to {SAVE_DIR}/tfidf_embeddings.npy")
    
    return features, categories, titles

if __name__ == "__main__":
    try:
        embedding, categories, titles = main()
        print("Embedding completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")