import logging
from pathlib import Path
from typing import Dict, Any
import fitz  # PyMuPDF for PDF analysis
from src.utils.ingestion.ingestion_metadata import TEXT_EXTENSIONS, IMAGE_EXTENSIONS, FileType, PDFType, IngestionMetadata

class SourceClassifier:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("src.utils.ingestion.source_classifier")

    def classify_file(self, file_path: Path) -> IngestionMetadata:
        """
        Classifies a file based on its extension and returns metadata for ingestion.
    
        This method examines the file extension of the provided file path and assigns
        appropriate metadata, including file type and support status. It is used to
        determine how a file should be processed in the ingestion pipeline (e.g., text
        extraction for PDFs or text files, OCR for images, or skipping unsupported files).
    
        Args:
            file_path (Path): The path to the file to classify, provided as a Path object.
    
        Returns:
            IngestionMetadata: An object containing metadata about the file, including:
                - file_name: The name of the file (e.g., 'document.pdf').
                - file_path: The full file path as a string.
                - file_type: An enum value (FileType.PDF, FileType.TXT, FileType.IMAGE, or FileType.UNKNOWN).
                - is_supported: A boolean indicating if the file type is supported (False for unknown types).
                - error: An error message if the file type is unsupported (None otherwise).
    
        Notes:
            - The function relies on predefined sets of extensions: `TEXT_EXTENSIONS` and `IMAGE_EXTENSIONS`.
            - If the file extension is not recognized, the file is marked as unsupported with an error message.
            - This method does not analyze file content; it only uses the file extension for classification.
        """
        ext = file_path.suffix.lower()
        meta = IngestionMetadata(file_name=file_path.name, file_path=str(file_path))
        if ext == ".pdf":
            meta.file_type = FileType.PDF
            self.classify_pdf_type(file_path)
        elif ext == ".p7m":
            meta.file_type = FileType.P7M
            self.classify_pdf_type(file_path) # Treat P7M like PDF for classification
        elif ext in TEXT_EXTENSIONS:
            meta.file_type = FileType.TXT
        elif ext in IMAGE_EXTENSIONS:
            meta.file_type = FileType.IMAGE
        else:
            meta.is_supported = False
            meta.file_type = FileType.UNKNOWN
            meta.error = "Unsupported file type."
        return meta

    def classify_pdf_type(self, file_path) -> PDFType:
        """
        Classifies a PDF file as text-based, image-based, or unknown based on its extractable text content.

        This method uses PyMuPDF (fitz) to analyze the text content of a PDF file. It calculates
        the average text length per page and classifies the PDF as text-based if it contains
        substantial extractable text (e.g., digitally created PDFs) or image-based if it has
        minimal text (e.g., scanned documents requiring OCR). If an error occurs during processing,
        the PDF is classified as unknown.

        Args:
            file_path (str or Path): The path to the PDF file to classify.

        Returns:
            PDFType: An enum value indicating the PDF type:
                - PDFType.TEXT_BASED: The PDF contains significant extractable text.
                - PDFType.IMAGE_BASED: The PDF contains minimal text, likely requiring OCR.
                - PDFType.UNKNOWN: The PDF could not be classified due to an error.

        Notes:
            - The classification is based on a threshold of characters per page on average.
              If the average text length per page exceeds the threshold, the PDF is considered text-based;
              otherwise, it is image-based.
            - The text is encoded in UTF-8 to handle special characters properly.
            - The document is closed in a `finally` block to ensure resources are freed, even if an error occurs.
        """
        try:
            doc = fitz.open(file_path)
            total_text_length = 0
            for page in doc:
                text = page.get_text().encode("utf8").strip()
                total_text_length += len(text)

            # Threshold: If average text per page is low, consider image_based
            avg_text = total_text_length / len(doc) if len(doc) > 0 else 0
            if avg_text > 80:
                return PDFType.TEXT_BASED
            else:
                return PDFType.IMAGE_BASED
        except Exception as e:
            self.logger.error("Error classifying PDF type for %s: %s", file_path, str(e))
            return PDFType.UNKNOWN
        # finally:
        #     doc.close()