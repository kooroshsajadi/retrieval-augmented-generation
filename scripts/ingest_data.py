import logging
from typing import Dict, List, Optional, Tuple, Any
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
from pathlib import Path
import fitz  # PyMuPDF
from src.utils.logging_utils import setup_logger
import os

class DataIngestor:
    """Extracts text from a single PDF, text file, or image for RAG pipeline."""

    def __init__(
        self,
        output_dir: str = "data/texts",
        max_pages: Optional[int] = None,
        language: str = "ita",
        tessdata_dir: Optional[str] = None,
        logger: logging.Logger = None
    ):
        """
        Initialize DataIngestor.

        Args:
            output_dir (str): Directory to save extracted text.
            max_pages (Optional[int]): Maximum pages to process for PDFs.
            language (str): Language for OCR (default: Italian).
            tessdata_dir (Optional[str]): Path to Tesseract OCR data.
            logger (logging.Logger, optional): Logger instance.
        """
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.language = language
        self.logger = logger or setup_logger("data_ingestor")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure Tesseract for OCR
        if tessdata_dir:
            self.tessdata_dir = tessdata_dir
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir

    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, bool]:
        """
        Extract text from a PDF file, trying text-based extraction first, then OCR if needed.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        # Try text-based extraction with pdfplumber
        try:
            self.logger.info(f"Attempting text-based extraction for PDF: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                pages_to_read = min(len(pdf.pages), self.max_pages) if self.max_pages else len(pdf.pages)
                text = ""
                for i in range(pages_to_read):
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    self.logger.info(f"Successfully extracted text from PDF: {file_path}")
                    return text, True
        except Exception as e:
            self.logger.warning(f"Text-based extraction failed for {file_path}: {e}")

        # Fallback to OCR if text-based extraction fails or yields no text
        self.logger.info(f"Falling back to OCR for PDF: {file_path}")
        try:
            images = convert_from_path(file_path, first_page=1, last_page=self.max_pages) if self.max_pages else convert_from_path(file_path)
            text = ""
            for i, image in enumerate(images):
                self.logger.debug(f"Running Tesseract OCR on page {i + 1}")
                image = ImageEnhance.Contrast(image).enhance(2.0)
                page_text = pytesseract.image_to_string(image, lang=self.language)
                text += page_text + "\n"
            success = len(text.strip()) > 0
            self.logger.info(f"OCR extraction {'successful' if success else 'failed'} for {file_path}")
            return text, success
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {file_path}: {e}")
            return "", False

    def extract_text_from_txt(self, file_path: Path) -> Tuple[str, bool]:
        """
        Extract text from a text file.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            success = len(text.strip()) > 0
            self.logger.info(f"Text file extraction {'successful' if success else 'failed'} for {file_path}")
            return text, success
        except Exception as e:
            self.logger.error(f"Failed to read text file {file_path}: {e}")
            return "", False

    def extract_text_from_image(self, file_path: Path) -> Tuple[str, bool]:
        """
        Extract text from an image file using Tesseract OCR.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        try:
            self.logger.info(f"Running Tesseract OCR on image: {file_path}")
            image = Image.open(file_path)
            image = ImageEnhance.Contrast(image).enhance(2.0)
            text = pytesseract.image_to_string(image, lang=self.language)
            success = len(text.strip()) > 0
            self.logger.info(f"Image extraction {'successful' if success else 'failed'} for {file_path}")
            return text, success
        except Exception as e:
            self.logger.error(f"OCR failed for image {file_path}: {e}")
            return "", False

    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a single file (PDF, text, or image).

        Args:
            file_path (str): Path to the input file.

        Returns:
            Dict[str, Any]: Dictionary with file details and extracted text.
        """
        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower().lstrip("."),
            "text": "",
            "is_valid": False,
            "error": None
        }

        self.logger.info(f"Processing file: {file_path}")

        if result["file_type"] in ["pdf"]:
            text, success = self.extract_text_from_pdf(file_path)
        elif result["file_type"] in ["txt"]:
            text, success = self.extract_text_from_txt(file_path)
        elif result["file_type"] in ["jpg", "jpeg", "png", "bmp"]:
            text, success = self.extract_text_from_image(file_path)
        else:
            result["error"] = f"Unsupported file type: {result['file_type']}"
            self.logger.error(result["error"])
            return result

        result["text"] = text
        result["is_valid"] = success
        if not success:
            result["error"] = f"No valid text extracted from {result['file_type']} file"

        # Save extracted text if valid
        if result["is_valid"]:
            self.save_extracted_text(file_path.stem, text)

        return result

    def save_extracted_text(self, file_base_name: str, text: str) -> None:
        """
        Save extracted text to output directory.

        Args:
            file_base_name (str): Base name of the file (without extension).
            text (str): Extracted text to save.
        """
        text_file = self.output_dir / f"{file_base_name}.txt"
        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            self.logger.info(f"Saved extracted text to {text_file}")
        except Exception as e:
            self.logger.error(f"Failed to save text to {text_file}: {e}")

if __name__ == "__main__":
    # Example usage
    ingestor = DataIngestor(
        output_dir="data/texts",
        max_pages=None,
        language="ita",
        tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata"  # Adjust as needed
    )
    result = ingestor.extract_text("data/source/sample.pdf")
    print(f"Extraction result: {result}")