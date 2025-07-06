# write a python script that
# 1. load all the pdfs in "data"
# 2. do a simple TF-IDF analysis on the text
# 3. do a tsne plot 
# 4. do a kmeans clustering

import os
import io
import json
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import google
import google.generativeai as genai
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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
    "ai_in_legal_proceedings": {
      "description": "AI usage and implications within court proceedings",
      "subcategories": {
        "general_use_of_ai_in_court": {
          "explanation": "Using AI to write the case, having AI made material to convince the jury"
        },
        "hallucinations_in_court": {},
        "testifying_expert": {
          "explanation": "Who counts as an expert for AI, can AI count as an expert?"
        },
        "court_and_attorney_rules": {
          "explanation": "You should only cite or submit things that you believe are true"
        }
      }
    },
    "antitrust": {
      "description": "Competition law and market regulation issues",
      "subcategories": {
        "fraudulent_market_practices": {"explanation": "Using AI for fraudulent market practices"},
        "price_fixing": {"explanation": "Price fixing either using or concerning AI"},
        "collusion": {"explanation": "Collusion between tech giants, or collusion surrounding AI"}
      }
    },
    "consumer_protection": {
      "description": "Laws protecting consumer rights and interests",
      "subcategories": {
        "fraudulent_market_practices": {"explanation": "Misinformation, harming the consumers through the use of AI"},
        "privacy_and_power": {"explanation": "Harming the consumer's rights to privacy through the use of AI or to gain data to train AI with"}
      }
    },
    "intellectual_property_law": {
      "description": "IP rights, patents, and related legal issues",
      "subcategories": {
        "fair_use": {},
        "privacy_and_power": {},
        "patent": {"explanation": "Concerned with if adding AI counts qualifies for a new patent"}
      }
    },
    "torts": {
        "description": "A wrongful act or an infringement of a right (other than under contract) leading to civil legal liability. Torts are common law causes of action",
        "note": "Use this category when the core dispute involves civil liability between private actors",
        "subcategories": {
            "ai_damage_to_humans": {
            "explanation": "Harm caused by AI systems to individuals, including hallucinations, defamation, emotional distress, or physical injury."
            },
            "product_liability_negligence": {
            "explanation": "Negligent design, testing, or failure to warn about defective products — including AI-based or tech-integrated systems."
            },
            "privacy_and_power": {
            "explanation": "Private-sector intrusion on individual privacy, such as corporate surveillance or unauthorized use of user data. Do not use if the harm arises from government action — use Justice and Equity instead."
            },
            "section_230": {
            "explanation": "Claims about platform responsibility for third-party or algorithmically surfaced content, especially involving defamation, discrimination, or user-generated material."
            }
        }
    },
    "justice_and_equity": {
      "description": "Cases involving constitutional rights, civil liberties, or government misconduct. Includes surveillance, discrimination, due process, or unlawful use of state power.",
      "note": "Use this when the case is against a government entity or raises issues of constitutional significance (e.g., freedom of speech, unlawful search, state bias). Do not use for civil disputes between private parties — those go under Torts.",
      "subcategories": {
        "constitutional_and_civil_rights": {
          "explanation": "Challenges to government action under the Fourth, First, or other constitutional amendments (e.g., surveillance, censorship, compelled speech)."
        },
        "bias_and_discrimination": {
          "explanation": "Unlawful or systemic bias, including race, gender, or other protected characteristics — especially where government or large platforms are involved."
        },
        "due_process": {
          "explanation": "Failures in procedural fairness, including secret algorithms or automated systems used in high-stakes decisions without transparency or appeal."
        }
      }
    },
    "privacy_data_protection": {
      "description": "Cases involving statutory privacy rights, consumer data use, or unauthorized data sharing by private entities. Focused on data governance, not general harm.",
      "note": "Use this category when plaintiffs invoke statutes like BIPA, CCPA, GDPR, or similar consumer privacy frameworks. Do NOT use if the privacy harm comes from government surveillance (use Justice and Equity) or if it's a tort-based dispute without statutory grounding (use Torts).",
      "subcategories": {
        "privacy_and_power": {
          "explanation": "Commercial exploitation of personal data, surveillance capitalism, or privacy-invasive platform practices."
        },
        "ai_damage_to_humans": {
          "explanation": "Loss of privacy resulting directly from AI systems misusing or revealing personal information."
        },
        "individual_rights_to_suit": {
          "explanation": "Right of individuals to bring claims under privacy statutes like BIPA or CCPA, including biometric consent or data sale violations."
        }
      }
    },
    "unrelated": {
      "description": "Cases where AI is does not pertain towards the legal issue at all. If the case is unrelated, the case should fit in no other category",
      "subcategories": {
        "testifying_expert": {
          "explanation": "Testifying expert has AI in description, but AI experience is not relevant towards the testifying content"
        },
        "ai_as_example": {
          "explanation": "The rest of the case does not ever mention or relate to anything towards AI. Reasoning using AI does not change anything."
        },
        "ai_mentioned_in_passing": {
          "explanation": "AI mentioned in passing but has no impact towards any logic present within the case"
        }
      }
    }
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
    """

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
    """
    # --- Categorization ---
    print("categorizing...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    case_categories = {}
    for name, text in texts.items():
        prompt = f"""
Categorize each case into one or more primary containers from the list below, but only when the legal issues at the heart of the case make those categories truly salient. 
Avoid tagging multiple categories unless necessary. A case should not be tagged as AI-related merely because AI is mentioned — only if it forms a central part of the legal dispute.
**Important Guidance for Categorization:**
If a case is not substantively about AI, place it in "Unrelated", even if the term appears in job titles, company names, or marketing material. 
If a case is Unrelated, it may not go into any other category.
{json.dumps(category_descriptions, indent=2)}

Only respond with the category name, nothing else.

Case content:
{text[:3000]}
"""
        response = model.generate_content(prompt)
        category = response.text.strip()
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