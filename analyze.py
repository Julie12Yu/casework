# write a python script that
# 1. load all the pdfs in "data"
# 2. do a simple TF-IDF analysis on the text
# 3. do a tsne plot 
# 4. do a kmeans clustering

import os
import io
import json
import subprocess
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import PyPDF2
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# --- Config ---
GOOGLE_DRIVE_FOLDER_NAME = "may_25_court_pdfs"
SCOPES = ["https://www.googleapis.com/auth/drive"]
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USE_LOCAL_ONLY = True

# --- Google OAuth User-Based Flow ---
def get_google_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def search_documents(service, query):
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return results.get('files', [])

def download_pdf(service, file_id, filename):
    filepath = DATA_DIR / filename
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(filepath, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()

# --- Download and Extract Text ---
def extract_texts():
    texts = defaultdict(str)
    for pdf_file in os.listdir(DATA_DIR):
        if pdf_file.endswith(".pdf"):
            with open(DATA_DIR / pdf_file, "rb") as file:
                pdf = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                texts[pdf_file] = text
    with open("texts.json", "w") as f:
        json.dump(texts, f)
    return texts

# --- Categorization Setup ---
category_descriptions = {
    "Privacy and power": "AI deriving intimate information from available data; legal doctrine gaps in privacy law coverage",
    "Justice and equity": "AI discrimination and differential treatment affecting demographic progress and opportunities",
    "Fair Use and Ownership": "Content ownership rights and permissible use of materials for AI training",
    "Section 230 and Platform Liability": "Responsibility and accountability for AI-generated content and outputs",
    "AI Court Misconduct": "Legal professionals caught using AI that produced hallucinated information in court proceedings",
    "AI Expert Testimony": "Cases where AI technology is relevant due to expert witnesses having AI backgrounds or expertise",
    "AI as Legal Example": "AI used as illustrative examples or supporting factors to clarify or strengthen legal arguments",
    "Fraudulent Market Practices": "AI used for market manipulation, trading fraud, or as shell companies for illegal activities",
    "Antitrust": "Competition issues between smaller AI companies and major tech corporations",
    "AI Human Harm": "Cases where AI systems are accused of causing direct physical or emotional damage to individuals",
    "Other": "Cases that don't fit into the above AI-related legal categories"
}

# --- Main Execution ---
def main():
    if not USE_LOCAL_ONLY:
        service = get_google_drive_service()
        folder_query = f"name='{GOOGLE_DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folders = search_documents(service, folder_query)
        if not folders:
            print("Folder not found.")
            return
        folder_id = folders[0]['id']

        pdf_query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        pdfs = search_documents(service, pdf_query)
        for pdf in pdfs[:50]:
            download_pdf(service, pdf['id'], pdf['name'])
    else:
        print("USING LOCAL FILES ONLY – skipping Drive download")

    texts = extract_texts()
    print("extracted texts!")
    docs = list(texts.values())
    print("got docs..")
    names = list(texts.keys())

    # --- TF-IDF + t-SNE ---
    print("starting on tf-idf and t-sne")
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)

    tsne = TSNE(n_components=2, random_state=42, init="random")
    tsne_result = tsne.fit_transform(tfidf.toarray())

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(tfidf)

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title("t-SNE with KMeans")
    plt.savefig("tsne_plot.png")

    # --- Categorization ---
    case_categories = {}
    for name, text in texts.items():
        prompt = f"""
Your max response is up to 4 words long. DO NOT respond with more than four words.
Only respond with the category name, nothing else.
You are classifying legal cases that involve artificial intelligence into categories.
Choose one of the following categories that best fits this case, based on the descriptions below:

{json.dumps(category_descriptions, indent=2)}

Only respond with the category name, nothing else.

Case content:
{text[:3000]}
"""
        with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
            tf.write(prompt)
            tf.flush()
            result = subprocess.run(
                ["ollama", "run", "llama3"],
                input=prompt,
                text=True,
                capture_output=True
            )
            if name == "some_example.pdf":
                print(prompt)
            category = result.stdout.strip()
        case_categories[name] = category

    with open("case_categories.json", "w") as f:
        json.dump(case_categories, f, indent=2)

    for doc, cat in case_categories.items():
        print(f"{doc}: {cat}")

        # --- Group by Category ---
    grouped_categories = defaultdict(list)
    for case_name, category in case_categories.items():
        grouped_categories[category].append(case_name)

    with open("grouped_case_categories.json", "w") as f:
        json.dump(grouped_categories, f, indent=2)


if __name__ == "__main__":
    main()
