import json
from typing import Dict, List, Optional, Tuple, Any
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
from pathlib import Path
import os
import fitz  # PyMuPDF
import logging
from src.utils.logging_utils import setup_logger
import yaml


class PyMuPDFImageOCR:
    """Perform OCR on standalone image files using PyMuPDF's built-in OCR support."""

    def __init__(self, tessdata_dir: Optional[str] = None, logger=None):
        self.tessdata_dir = tessdata_dir
        self.logger = logger or logging.getLogger("pymupdf_image_ocr")
        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir

    def ocr_image_file(self, image_path: Path) -> str:
        try:
            pix = fitz.Pixmap(str(image_path))
            doc = fitz.open()
            page = doc.new_page(width=pix.width, height=pix.height)
            page.insert_image(page.rect, pixmap=pix)

            textpage = page.get_textpage_ocr(dpi=300, full=True)
            text = page.get_text(textpage=textpage)

            doc.close()
            pix = None
            return text
        except Exception as e:
            self.logger.error(f"OCR failed for image {image_path}: {e}")
            return ""


class TextIngestor:
    """Extracts text from PDFs, text files, and images using appropriate methods."""

    def __init__(
        self,
        input_dir: str = "data/destination",
        metadata_path: str = "data/metadata_file.json",
        output_dir: str = "data/extracted_text",
        output_metadata_file: str = "data/extracted_text/overall_metadata.json",
        max_pages: Optional[int] = None,
        language: str = "ita",
        tessdata_dir: Optional[str] = None,
    ):
        self.input_dir = Path(input_dir)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_metadata_file = Path(output_metadata_file)
        self.max_pages = max_pages
        self.language = language
        self.logger = setup_logger("text_ingestor")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PyMuPDFImageOCR helper for image OCR
        self.image_ocr = PyMuPDFImageOCR(tessdata_dir=tessdata_dir, logger=self.logger)

    def load_classification_metadata(self) -> List[Dict[str, Any]]:
        if not self.metadata_path.exists():
            self.logger.error("Metadata file not found: %s", self.metadata_path)
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load metadata: %s", e)
            raise

    def extract_text_with_pdfplumber(self, file_path: Path) -> Tuple[str, List[Dict], bool]:
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
            self.logger.error("pdfplumber extraction failed for %s: %s", file_path, e)
            return "", [{"page_number": 0, "error": str(e)}], False

    def extract_text_with_ocr(self, file_path: Path) -> Tuple[str, List[Dict], bool]:
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
                image = ImageEnhance.Contrast(image).enhance(2.0)
                page_text = pytesseract.image_to_string(image, lang=self.language)
                text += page_text + "\n"
                page_metadata.append({
                    "page_number": i + 1,
                    "text_length": len(page_text),
                    "source": "tesseract"
                })
            success = len(text.strip()) > 0
            return text, page_metadata, success
        except Exception as e:
            self.logger.error("Tesseract OCR extraction failed for %s: %s", file_path, e)
            return "", [{"page_number": 0, "error": str(e)}], False

    def extract_text_from_txt(self, file_path: Path) -> Tuple[str, List[Dict], bool]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            page_metadata = [{
                "page_number": 1,
                "text_length": len(text),
                "source": "text_file"
            }]
            success = len(text.strip()) > 0
            return text, page_metadata, success
        except Exception as e:
            self.logger.error("Failed to read text file %s: %s", file_path, e)
            return "", [{"page_number": 0, "error": str(e)}], False

    def extract_text(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "text": "",
            "page_metadata": [],
            "is_valid": False,
            "error": None
        }

        self.logger.info("Extracting text from %s (type: %s)", file_path, file_type)

        if file_type == "pdf":
            try:
                metadata_list = self.load_classification_metadata()
                matching_meta = next((m for m in metadata_list if m["file_name"] == file_path.name), None)
                pdf_type = matching_meta.get("pdf_type", "unknown") if matching_meta else "unknown"
            except Exception as e:
                self.logger.warning("Failed to get pdf_type from metadata: %s", e)
                pdf_type = "unknown"

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
                result["error"] = "No valid text extracted from PDF"

        elif file_type == "text":
            text, page_metadata, success = self.extract_text_from_txt(file_path)
            result["text"] = text
            result["page_metadata"] = page_metadata
            result["is_valid"] = success
            if not success:
                result["error"] = "No valid text extracted from text file"

        elif file_type == "image":
            text = self.image_ocr.ocr_image_file(file_path)
            is_valid = bool(text.strip())
            result["text"] = text
            result["page_metadata"] = [{"page_number": 1, "text_length": len(text), "source": "pymupdf_ocr"}]
            result["is_valid"] = is_valid
            if not is_valid:
                result["error"] = "No valid text extracted from image file"
            self.logger.info(f"Extracted text from image file: {file_path} valid: {is_valid}")

        else:
            result["text"] = ""
            result["page_metadata"] = []
            result["is_valid"] = False
            result["error"] = f"Unsupported file type: {file_type}"
            self.logger.warning(f"Unsupported file type: {file_type}")

        return result

    def save_extracted_metadata(self, all_metadata: List[Dict[str, Any]]) -> None:
        try:
            with open(self.output_metadata_file, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved aggregated extraction metadata to {self.output_metadata_file}")
        except Exception as e:
            self.logger.error(f"Failed to save aggregated metadata to {self.output_metadata_file}: {e}")
            raise

    def save_extracted_text(self, file_name: str, text: str) -> None:
        text_file = self.output_dir / f"{file_name}.txt"
        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            self.logger.info(f"Saved extracted text to {text_file}")
        except Exception as e:
            self.logger.error(f"Failed to save text to {text_file}: {e}")

    def process_directory(self) -> None:
        metadata = self.load_classification_metadata()
        if not metadata:
            self.logger.warning("No classification metadata found. Skipping processing.")
            return

        files_metadata = {m["file_name"]: m for m in metadata}
        processed_files = 0
        aggregated_metadata = []

        for file_name, classification in files_metadata.items():
            file_path = self.input_dir / file_name
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                aggregated_metadata.append({
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "file_type": classification.get("file_type", "unknown"),
                    "text": "",
                    "page_metadata": [],
                    "is_valid": False,
                    "error": "File missing"
                })
                continue

            result = self.extract_text(file_path, classification.get("file_type", "unknown"))
            file_base_name = file_name.rsplit(".", 1)[0]
            if result["text"]:
                self.save_extracted_text(file_base_name, result["text"])
            else:
                self.logger.info(f"No text extracted to save for file: {file_name}")

            meta_entry = {
                "file_path": result["file_path"],
                "file_name": result["file_name"],
                "file_type": result.get("file_type", "unknown"),
                "page_metadata": result["page_metadata"],
                "is_valid": result["is_valid"],
                "error": result["error"]
            }
            aggregated_metadata.append(meta_entry)
            processed_files += 1

        self.save_extracted_metadata(aggregated_metadata)
        self.logger.info(f"Processed {processed_files}/{len(files_metadata)} files")


if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)

    ingestor = TextIngestor(
        input_dir=config['files']['prefettura_v1'], # Replace with your actual local address to the dataset.
        metadata_path=config['metadata']['classification_prefettura_v1'], # Replace with your local address to load the classification metadata.
        output_dir=config['texts']['prefettura_v1'], # Replace with you desired local address to save the extracted text.
        output_metadata_file=config['metadata'].get('extraction_prefettura_v1.2', 'data/metadata/extraction_prefettura_v1.2.json'), # Replace with your desired local address to save the extraction metadata.
        max_pages=None,
        language="ita",
        tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata"  # adjust as needed.
    )
    ingestor.process_directory()
    print("Text extraction completed. Aggregated metadata saved.")
