import requests
import re
import os
import time
from urllib.parse import urlparse, unquote
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import PyPDF2
from contextlib import contextmanager

# Configuration
API_KEY = "kqlpY3lEpu92JTjgNhdQyD56UkstzI0fhUsYNtYj"
SEARCH_QUERY = "artificial intelligence collection:USCOURTS"
PAGE_SIZE = 100  # Increase this for more results per request
MAX_RESULTS = None # Set to None for no limit
GOOGLE_DRIVE_FOLDER_NAME = "july_9_court_pdfs"  # Name of the folder to create in Google Drive

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_google_drive_service():
    """Set up and return the Google Drive API service"""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    # Return Drive API service
    return build('drive', 'v3', credentials=creds)

def create_drive_folder(service, folder_name):
    """Create a folder in Google Drive and return its ID"""
    # Check if folder already exists
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if items:
        # Folder exists, return its ID
        return items[0]['id']
    else:
        # Create new folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')

def search_documents(offset_mark="*"):
    """Get search results from the API using the proper POST request format"""
    url = "https://api.govinfo.gov/search"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Api-Key": API_KEY
    }
    
    payload = {
        "query": SEARCH_QUERY,
        "pageSize": PAGE_SIZE,
        "offsetMark": offset_mark,
        "sorts": [
            {
                "field": "relevancy",
                "sortOrder": "DESC"
            }
        ],
        "historical": True,
        "resultLevel": "default"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error searching documents: {response.status_code}")
        print(response.text)
        return None

def upload_pdf_to_drive(service, pdf_link, filename, folder_id):
    """Download a PDF from the provided link and upload to Google Drive"""
    headers = {"X-Api-Key": API_KEY}
    
    # Check if file already exists in the folder to avoid duplicates
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if items:
        print(f"File {filename} already exists in Drive (ID: {items[0]['id']}). Skipping upload.")
        return True
    
    # Stream the PDF content
    response = requests.get(pdf_link, headers=headers, stream=True)
    
    if response.status_code == 200:
        # Create file metadata
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        print(f"Processing file: " + filename)
        # Create an in-memory file-like object
        pdf_content = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            pdf_content.write(chunk)
        pdf_content.seek(0)
        # CEHCK IF PASSES FILTER
        print(f"CHECKING FOR FILTER")
        if not contains_ai_filter(pdf_content):
            print(f"Does not contain the word \"artificial intelligence\"")
            return False
        print("Passed, uploading to google drive now!")
        # Upload to Google Drive
        media = MediaIoBaseUpload(
            pdf_content, 
            mimetype='application/pdf',
            resumable=True
        )
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Uploaded: {filename} (Drive ID: {file.get('id')})")
        return True
    else:
        print(f"Error downloading {pdf_link}: {response.status_code}")
        print(response.text)
        return False

def contains_ai_filter(pdf_content):
    try:  # ADDED: Wrap in timeout
        pdf_content.seek(0)  # Reset position
        reader = PyPDF2.PdfReader(pdf_content)
        text = ""
        total = 0
        for page in reader.pages:
            if total > 10:
                break
            text += page.extract_text()
            total += 1
        """Check if text contains 'artificial intelligence' as a phrase, handling line breaks"""
        if not text:
            return False
        
        # Convert to lowercase and normalize whitespace/line breaks
        text_lower = text.lower()
        
        # Replace any whitespace (including line breaks, tabs, etc.) with single spaces
        # This handles cases where "artificial" and "intelligence" are split by line breaks
        normalized_text = re.sub(r'\s+', ' ', text_lower)
        
        # Remove common PDF artifacts that might interfere
        # Remove non-printable characters and weird Unicode
        cleaned_text = re.sub(r'[^\w\s\.,;:!?()"\'-]', ' ', normalized_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Re-normalize after cleaning
        
        # Check for exact phrase "artificial intelligence"
        if "artificial intelligence" in cleaned_text:
            return True
        
        # Also check for common variations and abbreviations
        ai_patterns = [
            r'\bartificial\s+intelligence\b',  # artificial intelligence with any whitespace
        ]
        
        for pattern in ai_patterns:
            if re.search(pattern, cleaned_text):
                return True
        print(f"Failed the check")
        return False
    except Exception as e:  # ADDED: Handle other errors
        print(f"Error processing PDF: {e}")
        return False

def main():
    # Set up Google Drive service
    service = get_google_drive_service()
    
    # Create or get folder in Google Drive
    folder_id = create_drive_folder(service, GOOGLE_DRIVE_FOLDER_NAME)
    print(f"Using Google Drive folder: {GOOGLE_DRIVE_FOLDER_NAME} (ID: {folder_id})")
    
    offset_mark = "*"  # This is the initial value for the first page
    uploaded_count = 0
    total_count = 0
    more_results = True
    
    while more_results and (MAX_RESULTS is None or uploaded_count < MAX_RESULTS):
        print(f"Fetching results with offset mark: {offset_mark}...")
        results = search_documents(offset_mark)
        
        if not results or "results" not in results or not results["results"]:
            print("No more results found.")
            more_results = False
            continue
        
        print(f"Found {len(results['results'])} results in this page.")
        
        # Debug - print the entire response structure to see what fields are available
        print("Response keys:", list(results.keys()))
        
        # Properly check for offsetMark - it might be at a different level or have a different name
        if "offsetMark" in results:
            new_offset_mark = results["offsetMark"]
            print(f"Found offsetMark: {new_offset_mark}")
        else:
            print("WARNING: 'offsetMark' not found in response. Available keys:", list(results.keys()))
            # Look for alternative pagination fields
            if "pagination" in results:
                print("Pagination info:", results["pagination"])
            
            # Default to None
            new_offset_mark = None
        
        # Process results even if we didn't find a offsetMark
        for item in results["results"]:
            if "download" in item and "pdfLink" in item["download"]:
                pdf_link = item["download"]["pdfLink"]
                
                # Create a filename using case information
                case_id = item.get("packageId", "unknown")
                date = item.get("dateIssued", "")
                # Create a safe filename
                title = item.get("title", "")
                safe_title = "".join(c if c.isalnum() or c in [' ', '.', '-'] else '_' for c in title)
                safe_title = safe_title[:100]  # Limit length
                
                filename = f"{date}_{case_id}_{safe_title}.pdf"
                
                # Upload the PDF to Google Drive
                success = upload_pdf_to_drive(service, pdf_link, filename, folder_id)
                if success:
                    uploaded_count += 1
                    total_count += 1
                    print(f"Progress: {uploaded_count}/{MAX_RESULTS if MAX_RESULTS else 'unlimited'}")
                else:
                    total_count += 1
                # Be respectful of the API rate limits
                time.sleep(1)
                
                # Check if we've reached the maximum
                if MAX_RESULTS is not None and uploaded_count >= MAX_RESULTS:
                    print(f"Reached maximum number of uploads ({MAX_RESULTS}).")
                    more_results = False
                    break
        
        # Handle pagination based on what we found
        if new_offset_mark:
            if new_offset_mark == offset_mark:
                print("WARNING: Received same offset mark as current. Exiting to prevent infinite loop.")
                more_results = False
            else:
                offset_mark = new_offset_mark
                print(f"Moving to next page with offset mark: {offset_mark}")
        else:
            # If we didn't find a offsetMark, check if we have explicit pagination info
            if "pagination" in results and "next" in results["pagination"]:
                next_page = results["pagination"]["next"]
                print(f"Using alternative pagination: {next_page}")
                # Extract offset mark from next page URL if available
                # This is a placeholder - implement based on actual API response
                offset_mark = next_page
            else:
                print("No pagination information found. This appears to be the last page.")
                more_results = False
    
    print(f"Upload complete. Uploaded {uploaded_count} PDFs to Google Drive folder '{GOOGLE_DRIVE_FOLDER_NAME}'.")

if __name__ == "__main__":
    main()