import os
import json
from typing import Dict, Optional, Tuple

import paddle
import pdfplumber
from paddleocr import PaddleOCR
from pathlib import Path
import logging
import re
from src.utils.logging_utils import setup_logger

class PDFClassifier:
    """Classifies PDFs as text-based or image-based for ingestion pipeline."""

    def __init__(
        self,
        input_dir: str = "data/destination",
        metadata_dir: str = "data/metadata",
        min_text_length: int = 100,
        ocr_sample_pages: int = 1,
        language: str = "ita",
    ):
        """
        Initialize PDFClassifier with configuration parameters.

        Args:
            input_dir (str): Directory containing PDFs to classify.
            metadata_dir (str): Directory to save classification metadata.
            min_text_length (int): Minimum character count to consider a PDF text-based.
            ocr_sample_pages (int): Number of pages to sample for OCR check.
            language (str): Language code for PaddleOCR (e.g., 'ita' for Italian).
        """
        self.input_dir = Path(input_dir)
        self.metadata_dir = Path(metadata_dir)
        self.min_text_length = min_text_length
        self.ocr_sample_pages = ocr_sample_pages
        self.language = language
        self.logger = setup_logger("pdf_classifier")
        self.paddle_ocr = None  # Lazy initialization for PaddleOCR
        self.ontology_terms = {
            "ex:Technology": r"\b(technology|tecnologia|AI|intelligenza artificiale)\b",
            "ex:Legislation": r"\b(legge|decreto|articolo|regulation)\b",
        }  # Example ontology terms for validation

        # Ensure metadata directory exists
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def initialize_paddle_ocr(self) -> None:
        """Initialize PaddleOCR instance if not already initialized."""
        if self.paddle_ocr is None:
            try:
                self.logger.info("Initializing PaddleOCR with language: %s", self.language)
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=self.language, show_log=False)
            except Exception as e:
                self.logger.error("Failed to initialize PaddleOCR: %s", str(e))
                raise RuntimeError(f"PaddleOCR initialization failed: {e}")

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
                self.logger.info("Extracting text from %d pages with pdfplumber: %s", pages_to_read, file_path)
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

    def extract_text_with_paddleocr(self, file_path: Path) -> Tuple[str, bool]:
        """
        Perform lightweight OCR check using PaddleOCR on sample pages.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            Tuple[str, bool]: Extracted text and success flag.
        """
        self.initialize_paddle_ocr()
        text = ""
        try:
            from pdf2image import convert_from_path
            self.logger.info("Converting %d pages to images for OCR: %s", self.ocr_sample_pages, file_path)
            images = convert_from_path(file_path, first_page=1, last_page=self.ocr_sample_pages)
            for i, image in enumerate(images):
                self.logger.debug("Running PaddleOCR on page %d", i + 1)
                result = self.paddle_ocr.ocr(image, cls=True)
                if result:
                    for line in result:
                        if line:  # Ensure line is not None
                            for word_info in line:
                                text += word_info[1][0] + " "  # Extract text from OCR result
                text += "\n"
            success = self.is_valid_text(text)
            return text, success
        except Exception as e:
            self.logger.error("PaddleOCR extraction failed for %s: %s", file_path, str(e))
            return "", False

    def classify_pdf(self, file_path: Path) -> Dict[str, any]:
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
            "text_sample": "",
            "is_valid": False,
            "error": None
        }

        # Step 1: Try pdfplumber for text-based PDFs
        self.logger.info("Classifying PDF: %s", file_path)
        text, success = self.extract_text_with_pdfplumber(file_path)
        if success:
            result["pdf_type"] = "text-based"
            result["text_sample"] = text[:500]  # Truncate for metadata
            result["is_valid"] = True
            self.logger.info("Classified as text-based: %s", file_path)
            return result

        # Step 2: Fall back to PaddleOCR for image-based PDFs
        text, success = self.extract_text_with_paddleocr(file_path)
        if success:
            result["pdf_type"] = "image-based"
            result["text_sample"] = text[:500]
            result["is_valid"] = True
            self.logger.info("Classified as image-based: %s", file_path)
        else:
            result["error"] = "No valid text extracted with pdfplumber or PaddleOCR"
            self.logger.warning("Classification failed for %s: %s", file_path, result["error"])

        return result

    def process_directory(self) -> None:
        """
        Process all PDFs in the input directory and save classification metadata.
        """
        metadata_file = self.metadata_dir / "classification_metadata.json"
        metadata = []

        if not self.input_dir.exists():
            self.logger.error("Input directory does not exist: %s", self.input_dir)
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        pdf_files = [f for f in self.input_dir.glob("*.pdf")]
        if not pdf_files:
            self.logger.warning("No PDF files found in %s", self.input_dir)
            return

        self.logger.info("Processing %d PDF files in %s", len(pdf_files), self.input_dir)
        for file_path in pdf_files:
            result = self.classify_pdf(file_path)
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
    print("This module is not intended to be run directly. Use it as part of the RAG pipeline.")