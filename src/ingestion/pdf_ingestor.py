import json
import os
from typing import Dict, List, Optional, Tuple, Any
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
from pathlib import Path
import logging
from src.utils.logging_utils import setup_logger
import yaml

class PDFIngestor:
    """Extracts text from PDFs using pdfplumber or Tesseract based on classification."""

    def __init__(
        self,
        input_dir: str = "data/destination",
        metadata_path: str = "data/metadata_file.json",
        output_dir: str = "data/extracted_text",
        max_pages: Optional[int] = None,
        language: str = "it",
    ):
        """
        Initialize PDFIngestor with configuration parameters.

        Args:
            input_dir (str): Directory containing PDFs to process.
            metadata_path (str): Path to the file containing classification metadata.
            output_dir (str): Directory to save extracted text files.
            max_pages (Optional[int]): Maximum pages to process per PDF (default: all).
            language (str): Language code for Tesseract OCR (e.g., 'it' for Italian).
        """
        self.input_dir = Path(input_dir)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.language = language
        self.logger = setup_logger("pdf_ingestor")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_classification_metadata(self) -> List[Dict[str, Any]]:
        """
        Load classification metadata from JSON file.

        Returns:
            List[Dict[str, Any]]: List of classification result dictionaries.
        """
        metadata_file = self.metadata_path
        if not metadata_file.exists():
            self.logger.error("Metadata file not found: %s", metadata_file)
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load metadata: %s", str(e))
            raise

    def extract_text_with_pdfplumber(self, file_path: Path) -> Tuple[str, List[Dict], bool]:
        """
        Extract text from a text-based PDF using pdfplumber.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, List[Dict], bool]: Extracted text, page metadata, and success flag.
        """
        text = ""
        page_metadata = []
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_to_read = min(len(pdf.pages), self.max_pages) if self.max_pages else len(pdf.pages)
                self.logger.info("Extracting text from %d pages with pdfplumber: %s", pages_to_read, file_path)
                for i in range(pages_to_read):
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        page_metadata.append({
                            "page_number": i + 1,
                            "text_length": len(page_text),
                            "source": "pdfplumber"
                        })
                    else:
                        page_metadata.append({
                            "page_number": i + 1,
                            "text_length": 0,
                            "source": "pdfplumber",
                            "error": "No text extracted"
                        })
                success = len(text.strip()) > 0
                return text, page_metadata, success
        except Exception as e:
            self.logger.error("pdfplumber extraction failed for %s: %s", file_path, str(e))
            return "", [{"page_number": 0, "error": str(e)}], False

    def extract_text_with_ocr(self, file_path: Path) -> Tuple[str, List[Dict], bool]:
        """
        Extract text from an image-based PDF using Tesseract OCR.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, List[Dict], bool]: Extracted text, page metadata, and success flag.
        """
        text = ""
        page_metadata = []
        try:
            self.logger.info("Converting pages to images for OCR: %s", file_path)
            if self.max_pages is not None:
                images = convert_from_path(file_path, first_page=1, last_page=self.max_pages)
                pages_to_read = min(len(images), self.max_pages)
            else:
                images = convert_from_path(file_path)
                pages_to_read = len(images)
            self.logger.info("Extracting text from %d pages with Tesseract OCR", pages_to_read)
            for i, image in enumerate(images):
                self.logger.debug("Running Tesseract OCR on page %d", i + 1)
                # Enhance contrast
                image = ImageEnhance.Contrast(image).enhance(2.0)
                page_text = pytesseract.image_to_string(image, lang="ita")
                text += page_text + "\n"
                page_metadata.append({
                    "page_number": i + 1,
                    "text_length": len(page_text),
                    "source": "tesseract"
                })
            success = len(text.strip()) > 0
            return text, page_metadata, success
        except Exception as e:
            self.logger.error("Tesseract OCR extraction failed for %s: %s", file_path, str(e))
            return "", [{"page_number": 0, "error": str(e)}], False

    def extract_text(self, file_path: Path, pdf_type: str) -> Dict[str, Any]:
        """
        Extract text from a PDF based on its classification.

        Args:
            file_path (Path): Path to the PDF file.
            pdf_type (str): PDF type ('text-based', 'image-based', 'unknown').

        Returns:
            Dict[str, Any]: Extraction result with text and metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "pdf_type": pdf_type,
            "text": "",
            "page_metadata": [],
            "is_valid": False,
            "error": None
        }

        self.logger.info("Extracting text from %s (type: %s)", file_path, pdf_type)
        if pdf_type == "text-based":
            text, page_metadata, success = self.extract_text_with_pdfplumber(file_path)
        elif pdf_type == "image-based":
            text, page_metadata, success = self.extract_text_with_ocr(file_path)
        else:
            result["error"] = f"Unknown PDF type: {pdf_type}"
            self.logger.error(result["error"])
            return result

        result["text"] = text
        result["page_metadata"] = page_metadata
        result["is_valid"] = success
        if not success:
            result["error"] = "No valid text extracted"
            self.logger.warning("Text extraction failed for %s", file_path)

        return result

    def save_extracted_text(self, result: Dict[str, Any]) -> None:
        """
        Save extracted text and metadata to output directory.

        Args:
            result (Dict[str, Any]): Extraction result with text and metadata.
        """
        file_name = result["file_name"].rsplit(".", 1)[0]
        text_file = self.output_dir / f"{file_name}.txt"
        metadata_file = self.output_dir / f"{file_name}_metadata.json"

        # Save text
        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            self.logger.info("Saved extracted text to %s", text_file)
        except Exception as e:
            self.logger.error("Failed to save text to %s: %s", text_file, str(e))
            result["error"] = str(e)

        # Save metadata (excluding text)
        metadata = {
            "file_path": result["file_path"],
            "file_name": result["file_name"],
            "pdf_type": result["pdf_type"],
            "page_metadata": result["page_metadata"],
            "is_valid": result["is_valid"],
            "error": result["error"]
        }
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved extraction metadata to %s", metadata_file)
        except Exception as e:
            self.logger.error("Failed to save metadata to %s: %s", metadata_file, str(e))
            result["error"] = str(e)

    def process_directory(self) -> None:
        """
        Process all PDFs in the input directory based on classification metadata.
        """
        metadata = self.load_classification_metadata()
        if not metadata:
            self.logger.warning("No classification metadata found. Skipping processing.")
            return

        pdf_files = {m["file_name"]: m for m in metadata}
        processed_files = 0

        for file_name, classification in pdf_files.items():
            file_path = self.input_dir / file_name
            if not file_path.exists():
                self.logger.warning("PDF file not found: %s", file_path)
                continue

            if not classification["is_valid"]:
                self.logger.warning("Skipping invalid PDF: %s (%s)", file_name, classification.get("error", "Unknown error"))
                continue

            result = self.extract_text(file_path, classification["pdf_type"])
            self.save_extracted_text(result)
            processed_files += 1

        self.logger.info("Processed %d/%d PDF files", processed_files, len(pdf_files))

    def get_extraction_results(self) -> List[Dict[str, Any]]:
        """
        Load extraction results from metadata files in output directory.

        Returns:
            List[Dict[str, Any]]: List of extraction result dictionaries.
        """
        results = []
        for metadata_file in self.output_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                self.logger.error("Failed to load metadata: %s", str(e))
        return results

if __name__ == "__main__":
    with open('src/data/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        ingestor = PDFIngestor(
            input_dir=config['files']['prefettura_v1'],
            metadata_path=config['metadata']['prefettura_v1'],
            output_dir=config['texts']['prefettura_v1'],
            max_pages=None,
            language="it"
        )
        ingestor.process_directory()
        print("Text extraction completed. Results saved to data/extracted_text")
    except Exception as e:
        print(f"Error during text extraction: {e}")