GOOGLE_DRIVE_FOLDER_NAME = "may_25_court_pdfs"

# extract the pdfs from the google drive folder
import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]

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

def search_documents(service, query):
    """Search for documents in the Google Drive folder"""
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return results.get('files', [])

def download_pdf(service, file_id, filename):
    """Download a PDF file from Google Drive"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create full path for the file
    filepath = os.path.join(data_dir, filename)
    
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(filepath, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")
    fh.close()

def main():
    """Main function to execute the script"""
    # Set up Google Drive service
    service = get_google_drive_service()
    
    # Search for documents in the Google Drive folder
    query = f"name='{GOOGLE_DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folder_results = search_documents(service, query)
    
    if not folder_results:
        print(f"No folder found with name: {GOOGLE_DRIVE_FOLDER_NAME}")
        return
    
    # Get the folder ID from the search results
    folder_id = folder_results[0]['id']
    print(f"Found folder '{GOOGLE_DRIVE_FOLDER_NAME}' with ID: {folder_id}")
    
    # Search for PDF files in the folder - using a more general query
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    pdf_results = search_documents(service, query)
    
    # Print number of PDF files found
    num_files = len(pdf_results)
    print(f"Found {num_files} PDF files in folder '{GOOGLE_DRIVE_FOLDER_NAME}'")
    
    if num_files > 0:
        print("\nFiles found:")
        for pdf in pdf_results:
            print(f"- {pdf['name']}")

    # Download each PDF file
    for pdf in pdf_results:
        file_id = pdf['id']
        filename = pdf['name']
        download_pdf(service, file_id, filename)

if __name__ == "__main__":
    main()

