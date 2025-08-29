import os
import json
import sys
from pathlib import Path
from PyPDF2 import PdfReader
# How to use:
# python pdfs_to_json.py /path/to/pdf_directory
# It will write to [pdf_directory].json in the current directory.


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def convert_pdfs_to_json(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        print(f"Provided path is not a directory: {directory}")
        return

    pdf_texts = {}

    for file in directory.glob("*.pdf"):
        print(f"Processing: {file.name}")
        text = extract_text_from_pdf(file)
        pdf_texts[file.name] = text
    
    # Set output filename to <dirname>.json
    output_json = f"{directory.name}.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pdf_texts, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Output written to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdfs_to_json.py /path/to/pdf_directory")
    else:
        convert_pdfs_to_json(sys.argv[1])
