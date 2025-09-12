import logging
from pathlib import Path
from typing import Dict, Any

from src.utils.ingestion_metadata import TEXT_EXTENSIONS, IMAGE_EXTENSIONS, FileType

class SourceClassifier:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("source_classifier")

    def classify_file(self, file_path: Path) -> Dict[str, Any]:
        ext = file_path.suffix.lower()
        result = {
            "is_supported": True,
            "file_type": None,
            "pdf_type": None,
            "error": None,
            "file_name": file_path.name,
            "file_path": str(file_path),
        }
        if ext == ".pdf":
            result["file_type"] = FileType.PDF.value
            # Defer type determination until extraction
            result["pdf_type"] = None
        elif ext in TEXT_EXTENSIONS:
            result["file_type"] = FileType.TXT.value
        elif ext in IMAGE_EXTENSIONS:
            result["file_type"] = FileType.IMAGE.value
        else:
            result["is_supported"] = False
            result["file_type"] = FileType.UNKNOWN.value
            result["error"] = "Unsupported file type."
        return result
