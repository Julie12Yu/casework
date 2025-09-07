import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os

# File paths
DIR_PATH = 'court_pdfs_text.json'  # JSON mapping case name -> full text
SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Legal-BERT model & tokenizer
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_and_process_data():
    """
    Load JSON data (case_name -> full_text) and return dataframe + processed list
    """
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_cases = []
    for casename, full_text in data.items():
        case_info = {
            'title': casename,
            'text': full_text if isinstance(full_text, str) else "",
        }
        processed_cases.append(case_info)

    df = pd.DataFrame(processed_cases)
    return df, processed_cases

def create_legalbert_embeddings(texts, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(
            batch_texts, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # Mean pooling
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
    texts = [case['text'] for case in processed_cases]

    print(f"Loaded {len(titles)} cases")

    # Create Legal-BERT embeddings
    print("Creating Legal-BERT embeddings...")
    features = create_legalbert_embeddings(texts)
    print(f"Created embeddings of shape {features.shape}")

    # Save embeddings
    np.save(os.path.join(SAVE_DIR, "legalbert_embeddings.npy"), features)

    return features, titles

if __name__ == "__main__":
    try:
        embeddings, titles = main()
        print("Legal-BERT embedding pipeline completed successfully!")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")