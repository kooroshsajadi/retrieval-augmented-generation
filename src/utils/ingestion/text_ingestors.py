import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import fitz # PyMuPDF
from pathlib import Path
from typing import Tuple, List, Dict

def extract_text_with_pdfplumber(file_path: Path) -> Tuple[str, List[Dict], bool]:
    """
    Extract text from a PDF file using pdfplumber, with metadata for each page.

    Args:
        file_path (Path): Path to the PDF file.

    Returns:
        Tuple[str, List[Dict], bool]: Extracted text, list of page metadata, and success flag.
    """
    text, page_metadata = "", []
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_to_read = len(pdf.pages)
            for i in range(pages_to_read):
                page_text = pdf.pages[i].extract_text() or ""  # Handle None case
                text_length = len(page_text)
                text += page_text + "\n"
                page_metadata.append({"page_number": i + 1, "text_length": text_length})
        success = len(text.strip()) > 0
        return text, page_metadata, success
    except Exception as e: # TODO: Add error logging
        return "", [], False  # Empty metadata list for errors

def extract_text_with_ocr(file_path: Path, language: str = 'ita') -> Tuple[str, List[Dict], bool]:
    """
    Extract text from a PDF file using OCR, with metadata for each page.

    Args:
        file_path (Path): Path to the PDF file.
        language (str): Language for OCR (default: 'ita').

    Returns:
        Tuple[str, List[Dict], bool]: Extracted text, list of page metadata, and success flag.
    """
    text, page_metadata = "", []
    try:
        images = convert_from_path(pdf_path=file_path, dpi=300, first_page=1, use_pdftocairo=True)
        for i, image in enumerate(images):
            image = ImageEnhance.Contrast(image).enhance(2.0)
            page_text = pytesseract.image_to_string(image, lang=language) or ""
            text_length = len(page_text)
            text += page_text + "\n"
            page_metadata.append({"page_number": i + 1, "text_length": text_length})
        success = len(text.strip()) > 0
        return text, page_metadata, success
    except Exception as e:
        return "", [], False

def extract_text_from_txt(file_path: Path) -> Tuple[str, List[Dict], bool]:
    """
    Extract text from a text file, treating it as a single page.

    Args:
        file_path (Path): Path to the text file.

    Returns:
        Tuple[str, List[Dict], bool]: Extracted text, list of page metadata, and success flag.
    """
    try:
        text = Path(file_path).read_text(encoding="utf-8")
        text_length = len(text)
        return text, [{"page_number": 1, "text_length": text_length}], text_length > 0
    except (FileNotFoundError, UnicodeDecodeError) as e:
        return "", [], False

def extract_text_image_with_pymupdf(file_path: Path) -> Tuple[str, List[Dict], bool]:
    """
    Extract text from an image file using PyMuPDF with OCR.

    Args:
        file_path (Path): Path to the image file.

    Returns:
        Tuple[str, List[Dict], bool]: Extracted text, list of page metadata, and success flag.
    """
    try:
        pix = fitz.Pixmap(str(file_path))
        doc = fitz.open()
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        textpage = page.get_textpage_ocr(dpi=300, full=True)
        text = page.get_text(textpage=textpage)
        text_length = len(text)
        doc.close()
        pix = None
        return text, [{"page_number": 1, "text_length": text_length}], text_length > 0
    except (fitz.FileDataError, fitz.EmptyFileError) as e: # TODO: Add error logging
        return "", [], False
