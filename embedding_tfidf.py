import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
    """Load JSON data and process it for UMAP visualization"""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Process the new format
    processed_cases = []
    total = 0
    for casename, metadata_string in data.items():
        total += 1
        metadata = parse_case_metadata(metadata_string, total)
        
        # Extract main category
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

def create_text_features(summaries):
    """Create TF-IDF features from case summaries (titles in this case)"""
    # Clean and prepare text for TF-IDF
    cleaned_summaries = []    
    vectorizer = TfidfVectorizer(
        max_features=500,  # Reduced since we're working with case names
        stop_words=None,
        ngram_range=(1, 3),  # Include more n-grams for case names
        min_df=1,  # Allow single occurrences
        max_df=0.9
    )
    
    try:
        tfidf_features = vectorizer.fit_transform(cleaned_summaries)
        return tfidf_features.toarray(), vectorizer
    except ValueError as e:
        print(f"Error creating TF-IDF features: {e}")
        # Fallback: create simple character-based features
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
    
    # Extract data for processing
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]
    categories = [case['main_category'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create text features
    print("Creating TF-IDF features from case names...")
    features, vectorizer = create_text_features(summaries)
    print(f"Created {features.shape[1]} features")
    
    # Save TF-IDF embeddings
    np.save(os.path.join(SAVE_DIR, "tfidf_embeddings.npy"), features)
    pd.DataFrame(features, index=titles).to_csv(os.path.join(SAVE_DIR, "tfidf_embeddings.csv"))
    print(f"TF-IDF embeddings saved to {SAVE_DIR}/tfidf_embeddings.(npy/csv)")
    
    return features, categories, titles

if __name__ == "__main__":
    try:
        embedding, categories, titles = main()
        print("Interactive UMAP visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")