import json
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

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

def main():
    """Main function to run the embedding pipeline"""
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    titles = [case['title'] for case in processed_cases]
    texts = [case['text'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create embeddings with SentenceTransformers
     # Create text features
    print("Creating TF-IDF features from case names...")
    features, vectorizer = create_text_features(texts)
    print(f"Created {features.shape[1]} features")
    
    # Save TF-IDF embeddings
    np.save(os.path.join(SAVE_DIR, "tfidf_embeddings.npy"), features)
    print(f"TF-IDF embeddings saved to {SAVE_DIR}/tfidf_embeddings.npy")
    
    return features, titles

if __name__ == "__main__":
    try:
        embeddings, titles = main()
        print("Embedding generation completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")
