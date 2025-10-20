import json
from pathlib import Path
from src.utils.ingestion.ingestion_metadata import IngestionMetadata, FileType, PDFType
from src.utils.ingestion.source_classifier import SourceClassifier
import src.utils.ingestion.text_ingestors as ext
from typing import Dict, List, Optional
from src.utils.logging_utils import setup_logger
import logging


class TextExtractor:
    def __init__(self, input_dir,
                 output_dir,
                 output_metadata_file,
                 language="ita",
                 logger:Optional[logging.Logger]=None):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_metadata_file = Path(output_metadata_file)
        self.classifier = SourceClassifier()
        self.language = language
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("src.ingestion.text_extractor")
        # Dispatch table for file type to extraction function
        self._extraction_methods = {
            FileType.PDF: self._extract_pdf_wrapper,
            FileType.P7M: self._extract_pdf_wrapper,  # P7M treated like PDF
            FileType.TXT: self._extract_text_wrapper,
            FileType.IMAGE: self._extract_image_wrapper,
        }

    def _extract_pdf_wrapper(self, file_path: Path, meta: IngestionMetadata) -> tuple[str, List, bool]:
        """Extract text from a PDF file based on its pdf_type."""
        meta.pdf_type = self.classifier.classify_pdf_type(file_path)
        if meta.pdf_type == PDFType.TEXT_BASED:
            return ext.extract_text_with_pdfplumber(file_path)
        return ext.extract_text_with_ocr(file_path, language=self.language)
    
    def _extract_text_wrapper(self, file_path: Path, meta: IngestionMetadata) -> tuple[str, List, bool]:
        """Extract text from a text file."""
        return ext.extract_text_from_txt(file_path)
    
    def _extract_image_wrapper(self, file_path: Path, meta: IngestionMetadata) -> tuple[str, List, bool]:
        """Extract text from an image file."""
        return ext.extract_text_image_with_pymupdf(file_path)

    def _save_extracted_text(self, text: str, file_path: Path, meta: IngestionMetadata) -> None:
        """Save extracted text to file if valid and non-empty."""
        if meta.is_valid and text and text.strip():
            out_path = self.output_dir / (file_path.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            self.logger.info(f"Text extraction was successful for {file_path.name}")
        elif meta.is_valid and (not text or not text.strip()):
            self.logger.warning(f"No text extracted from {file_path.name} but not marked as invalid.")

    def _convert_metadata_to_dict(self, meta: IngestionMetadata) -> Dict:
        """Convert metadata object to dictionary, converting enums to their values."""
        meta_dict = meta.__dict__.copy()
        if isinstance(meta_dict.get("file_type"), FileType):
            meta_dict["file_type"] = meta_dict["file_type"].value
        if isinstance(meta_dict.get("pdf_type"), PDFType):
            meta_dict["pdf_type"] = meta_dict["pdf_type"].value
        return meta_dict

    def process_directory(self):
        """
        Process all files in the input directory, classify them, extract text, and save metadata.

        Iterates through files in the input directory, classifies each file using the classifier,
        extracts text based on file type, saves the extracted text to the output directory,
        and stores metadata in a JSON file. Unsupported files are logged and included in metadata
        with an error message.

        Attributes:
            input_dir (Path): Directory containing input files to process.
            output_dir (Path): Directory to save extracted text files.
            output_metadata_file (Path): Path to save the aggregated metadata JSON.
            language (str): Language for OCR (used for image-based PDFs).
            classifier: Object with methods to classify file types and PDF types.
            logger: Logger instance for logging info, warnings, and errors.

        Notes:
            - Supported file types are PDF (.pdf, .p7m), text files (e.g., .txt), and images (e.g., .png).
            - PDFs are further classified as text-based or image-based to determine extraction method.
            - Metadata is saved as JSON with enum values converted to strings.
            - Empty or invalid text extractions are logged but do not halt processing.
        """
        aggregated_metadata = []
        for file_path in self.input_dir.iterdir():
            if not file_path.is_file():
                continue

            # Classify file and initialize metadata
            cls_meta = self.classifier.classify_file(file_path)
            meta = IngestionMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=cls_meta.file_type,
                is_supported=cls_meta.is_supported,
                error=cls_meta.error,
                page_metadata=[],
                is_valid=True if cls_meta.is_supported else False
            )

            # Handle unsupported files
            if not meta.is_supported:
                self.logger.warning("Unsupported file type for %s", file_path.name)
            else:
                # Extract text using dispatch table
                extraction_method = self._extraction_methods.get(meta.file_type)
                if extraction_method:
                    text, meta.page_metadata, meta.is_valid = extraction_method(file_path, meta)
                    if not meta.is_valid:
                        self.logger.error("Text extraction failed for %s", file_path.name)
                    self._save_extracted_text(text, file_path, meta)
                else:
                    self.logger.error("No extraction method for file type %s", meta.file_type)

            # Convert and store metadata
            aggregated_metadata.append(self._convert_metadata_to_dict(meta))

        # Save aggregated metadata to JSON
        with open(self.output_metadata_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extractor = TextExtractor(
        input_dir=Path("data/prefettura_1_files"),
        output_dir=Path("data/prefettura_1_texts"),
        output_metadata_file=Path("data/metadata/extraction_prefettura_1.json"),
        language='ita'
    )
    extractor.process_directory()
    print("Text extraction and classification completed. Metadata saved.")
