import os
import numpy as np
import json
from typing import Dict, Optional, Tuple, Any
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from pathlib import Path
import logging
import re
from src.utils.logging_utils import setup_logger
from pdf2image import convert_from_path
import yaml


class SourceClassifier:
    """Classifies PDF, text, and image files for ingestion pipeline metadata."""

    TEXT_EXTENSIONS = {'.txt'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.tiff', '.gif'}

    def __init__(
        self,
        input_dir: str = "data/destination",
        metadata_dir: str = "data/metadata",
        min_text_length: int = 100,
        ocr_sample_pages: int = 1,
        language: str = "it",
    ):
        """
        Initialize SourceClassifier with configuration parameters.

        Args:
            input_dir (str): Directory containing files to classify.
            metadata_dir (str): Directory to save classification metadata.
            min_text_length (int): Minimum character count to consider text valid.
            ocr_sample_pages (int): Number of pages to sample for OCR check.
            language (str): Language code for OCR (e.g., 'it' for Italian).
        """
        self.input_dir = Path(input_dir)
        self.metadata_dir = Path(metadata_dir)
        self.min_text_length = min_text_length
        self.ocr_sample_pages = ocr_sample_pages
        self.language = language
        self.logger = setup_logger("source_classifier")
        self.ontology_terms = {
            "ex:Technology": r"\b(technology|tecnologia|AI|intelligenza artificiale)\b",
            "ex:Legislation": r"\b(legge|decreto|articolo|regulation)\b",
        }  # Example ontology terms for validation

        # Ensure metadata directory exists
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_text(self, text: str) -> bool:
        """
        Validate if extracted text is meaningful based on length and ontology terms.

        Args:
            text (str): Extracted text to validate.

        Returns:
            bool: True if text is valid (sufficient length or contains ontology terms).
        """
        if not text or len(text.strip()) < self.min_text_length:
            return False

        # Check for ontology terms (case-insensitive)
        for term, pattern in self.ontology_terms.items():
            if re.search(pattern, text, re.IGNORECASE):
                self.logger.debug("Found ontology term '%s' in text", term)
                return True

        # Check for Italian diacritics to confirm language relevance
        if re.search(r'[àèìòù]', text, re.IGNORECASE):
            self.logger.debug("Found Italian diacritics in text")
            return True

        # Fallback: Check word count (at least 10 words)
        words = text.split()
        return len(words) >= 10

    def extract_text_with_pdfplumber(self, file_path: Path) -> Tuple[str, bool]:
        """
        Attempt to extract text using pdfplumber.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_to_read = min(len(pdf.pages), self.ocr_sample_pages)
                self.logger.info(
                    "Extracting text from %d pages with pdfplumber: %s",
                    pages_to_read,
                    file_path,
                )
                for i in range(pages_to_read):
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                success = self.is_valid_text(text)
                return text, success
        except Exception as e:
            self.logger.error("pdfplumber extraction failed for %s: %s", file_path, str(e))
            return "", False

    def extract_text_with_ocr(self, file_path: Path) -> Tuple[str, bool]:
        """
        Perform lightweight OCR check using Tesseract on sample pages of PDF.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        text = ""
        try:
            self.logger.info(
                "Converting %d pages to images for OCR: %s", self.ocr_sample_pages, file_path
            )
            images = convert_from_path(file_path, first_page=1, last_page=self.ocr_sample_pages)
            for i, image in enumerate(images):
                self.logger.debug("Running Tesseract OCR on page %d", i + 1)
                # Enhance contrast
                image = ImageEnhance.Contrast(image).enhance(2.0)
                # Run Tesseract with Italian language
                page_text = pytesseract.image_to_string(image, lang="ita")
                text += page_text + "\n"
            success = self.is_valid_text(text)
            self.logger.debug("OCR extracted text (first 500 chars): %s", text[:500])
            self.logger.debug("Validation result: %s", success)
            return text, success
        except Exception as e:
            self.logger.error("Tesseract OCR extraction failed for %s: %s", file_path, str(e))
            return "", False

    def extract_text_from_txt(self, file_path: Path) -> Tuple[str, bool]:
        """
        Read text from a text file and validate.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            success = self.is_valid_text(text)
            return text, success
        except Exception as e:
            self.logger.error("Failed to read text file %s: %s", file_path, str(e))
            return "", False

    def classify_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Classify a PDF as text-based or image-based.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Dict[str, any]: Classification result with metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "pdf_type": "unknown",
            "is_valid": False,
            "error": None,
            "file_type": "pdf",
        }

        # Step 1: Try pdfplumber for text-based PDFs
        self.logger.info("Classifying PDF: %s", file_path)
        text, success = self.extract_text_with_pdfplumber(file_path)
        if success:
            result["pdf_type"] = "text-based"
            result["is_valid"] = True
            self.logger.info("Classified as text-based: %s", file_path)
            return result

        # Step 2: Fall back to Tesseract OCR for image-based PDFs
        text, success = self.extract_text_with_ocr(file_path)
        if success:
            result["pdf_type"] = "image-based"
            result["is_valid"] = True
            self.logger.info("Classified as image-based: %s", file_path)
        else:
            result["error"] = "No valid text extracted with pdfplumber or Tesseract OCR"
            self.logger.warning("Classification failed for %s: %s", file_path, result["error"])

        return result

    def classify_txt(self, file_path: Path) -> Dict[str, Any]:
        """
        Classify a text file by validating if it contains meaningful text.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            Dict[str, any]: Classification result with metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "is_valid": False,
            "error": None,
            "file_type": "text",
        }

        text, success = self.extract_text_from_txt(file_path)
        if success:
            result["is_valid"] = True
            self.logger.info("Valid text file: %s", file_path)
        else:
            result["error"] = "Text file content did not meet validation criteria"
            self.logger.warning("Invalid text file: %s", file_path)

        return result

    def classify_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Classify an image file by checking if it can be opened and is valid.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            Dict[str, any]: Classification result with metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "is_valid": False,
            "error": None,
            "file_type": "image",
        }

        try:
            with Image.open(file_path) as img:
                img.verify()  # Verifies that this is an image
            result["is_valid"] = True
            self.logger.info("Valid image file: %s", file_path)
        except Exception as e:
            result["error"] = f"Invalid image file: {e}"
            self.logger.warning("Invalid image file: %s, error: %s", file_path, e)

        return result

    def classify_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Determine the file type and classify accordingly.

        Args:
            file_path (Path): Path to the file.

        Returns:
            Dict[str, any]: Classification metadata dictionary.
        """
        ext = file_path.suffix.lower()
        self.logger.debug("Classifying file: %s with extension %s", file_path, ext)

        if ext == ".pdf":
            return self.classify_pdf(file_path)
        elif ext in self.TEXT_EXTENSIONS:
            return self.classify_txt(file_path)
        elif ext in self.IMAGE_EXTENSIONS:
            return self.classify_image(file_path)
        else:
            self.logger.warning("Unsupported file type: %s", file_path)
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "is_valid": False,
                "error": "Unsupported file type",
                "file_type": "unknown",
            }

    def process_directory(self) -> None:
        """
        Process all files in the input directory and save classification metadata.
        """
        metadata_file = self.metadata_dir / "classification_metadata.json"
        metadata = []

        if not self.input_dir.exists():
            self.logger.error("Input directory does not exist: %s", self.input_dir)
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files = [f for f in self.input_dir.iterdir() if f.is_file()]
        if not files:
            self.logger.warning("No files found in %s", self.input_dir)
            return

        self.logger.info("Processing %d files in %s", len(files), self.input_dir)
        for file_path in files:
            result = self.classify_file(file_path)
            metadata.append(result)

        # Save metadata to JSON
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved classification metadata to %s", metadata_file)
        except Exception as e:
            self.logger.error("Failed to save metadata: %s", str(e))
            raise

    def get_classification_results(self) -> list:
        """
        Load classification results from metadata file.

        Returns:
            list: List of classification result dictionaries.
        """
        metadata_file = self.metadata_dir / "classification_metadata.json"
        if not metadata_file.exists():
            self.logger.warning("Metadata file not found: %s", metadata_file)
            return []

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load metadata: %s", str(e))
            return []


if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        classifier = SourceClassifier(
            input_dir=config['files']['prefettura_v1'],
            metadata_dir=config['metadata']['directory'],
            min_text_length=100,
            ocr_sample_pages=1,
            language="it"
        )
        classifier.process_directory()
        print("Source classification completed.")
    except Exception as e:
        print(f"Error during classification: {e}")
