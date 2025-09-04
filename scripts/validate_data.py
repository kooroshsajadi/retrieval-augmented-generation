from pathlib import Path
import logging
from typing import Dict, Any, Optional
from src.utils.logging_utils import setup_logger

class DataValidator:
    """Validates user-provided files for supported formats and readability."""

    def __init__(
        self,
        supported_formats: list = [".text", ".txt", ".jpg", ".jpeg", ".gif", ".png", ".pdf"],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DataValidator.

        Args:
            supported_formats (list): List of supported file extensions (with dot).
            logger (logging.Logger, optional): Logger instance.
        """
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.logger = logger or setup_logger("data_validator")

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single file for existence, format, and readability.

        Args:
            file_path (str): Path to the file to validate.

        Returns:
            Dict[str, Any]: Validation result with status and details.
        """
        file_path = Path(file_path)
        result = {
            "file_path": file_path.as_posix(),
            "file_name": file_path.name,
            "is_valid": False,
            "error": None
        }

        self.logger.info("Validating file: %s", file_path)

        # Check if file exists
        if not file_path.exists():
            result["error"] = f"File not found: {file_path}"
            self.logger.error(result["error"])
            return result

        # Check file extension
        extension = file_path.suffix.lower()
        if extension not in self.supported_formats:
            result["error"] = f"Unsupported file format: {extension}. Supported formats: {', '.join(self.supported_formats)}"
            self.logger.error(result["error"])
            return result

        # Check if file is readable and not empty
        try:
            if extension in [".text", ".txt"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        result["error"] = "File is empty"
                        self.logger.error(result["error"])
                        return result
            elif extension in [".jpg", ".jpeg", ".gif", ".png"]:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
            elif extension == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    if not pdf.pages:
                        result["error"] = "PDF is empty"
                        self.logger.error(result["error"])
                        return result
            result["is_valid"] = True
            self.logger.info("File validated successfully: %s", file_path)
        except Exception as e:
            result["error"] = f"File is unreadable: {str(e)}"
            self.logger.error(result["error"])

        return result

if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    file_path = "data/source/sample.pdf"
    result = validator.validate_file(file_path)
    print(f"Validation result: {result}")