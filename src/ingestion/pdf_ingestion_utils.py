from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

class PDFIngestionUtils:
    def __init__(self, dpi: int = 300, lang: str = 'eng'):
        """
        Initializes the PDF OCR ingestion utility.
        
        Args:
            dpi (int): Resolution for image rendering from PDF.
            lang (str): Language code for OCR (Tesseract supported, e.g., 'eng', 'ita').
        """
        self.dpi = dpi
        self.lang = lang
        
    def extract_text_from_scanned_pdf(self, pdf_path: str) -> list:
        """
        Extract text from a scanned PDF using OCR.

        Args:
            pdf_path (str): Path to the scanned PDF file.

        Returns:
            List[str]: List of strings, where each entry is the OCR-extracted text from a page.
        """
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF file '{pdf_path}': {e}")

        text_pages = []
        for idx, page in enumerate(pages):
            try:
                text = pytesseract.image_to_string(page, lang=self.lang)
                text_pages.append(text)
            except Exception as ocr_error:
                print(f"[WARNING] OCR failed on page {idx} of '{pdf_path}': {ocr_error}")
                text_pages.append("")

        return text_pages
