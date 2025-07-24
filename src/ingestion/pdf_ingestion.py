import os
from src.ingestion.pdf_ingestion_utils import PDFIngestionUtils

DATA_DIR = "data/samples"  # Replace if needed
LANGUAGE = "ita"            # Tesseract language (e.g. 'eng', 'ita')

def ingest_documents():
    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith('.pdf')
    ]

    if not pdf_files:
        print("[INFO] No PDF files found.")
        return

    ingestion_util = PDFIngestionUtils(lang=LANGUAGE)

    for filepath in pdf_files:
        print(f"\n[INFO] Processing: {os.path.basename(filepath)}")
        try:
            text_pages = ingestion_util.extract_text_from_scanned_pdf(filepath)
            for i, page_text in enumerate(text_pages):
                print(f"\n--- Page {i+1} ---\n{text_pages[i][:]}")  # Print first N chars
        except Exception as e:
            print(f"[ERROR] Could not process {filepath}: {e}")

if __name__ == "__main__":
    ingest_documents()
