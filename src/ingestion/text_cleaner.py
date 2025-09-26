import json
import re
from typing import Dict, Any
from pathlib import Path
import yaml
from src.utils.logging_utils import setup_logger
from src.utils.deduplication_utils import fingerprint


class TextCleaner:
    """Cleans extracted text from PDFs for downstream processing."""

    def __init__(
        self,
        input_dir: str = "data/extracted_text",
        output_dir: str = "data/cleaned_text",
        min_text_length: int = 20,
    ):
        """
        Initialize TextCleaner with configuration parameters.

        Args:
            input_dir (str): Directory containing extracted text files.
            output_dir (str): Directory to save cleaned text.
            min_text_length (int): Minimum character count for valid cleaned text.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_text_length = min_text_length
        self.logger = setup_logger("text_cleaner")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Patterns for cleaning
        self.header_footer_patterns = [
            # r"Page\s+\d+\s*(of\s+\d+)?",  # e.g., "Page 1", "Page 1 of 5"
            # r"\d{1,2}/\d{1,2}/\d{2,4}",  # e.g., "12/31/2023"
            # r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?",  # e.g., "12:30", "12:30:45 PM"
        ]
        self.diacritics = r"[^\w\sàèìòùÀÈÌÒÙ.,:-;'\"“”‘’()]"

    def clean_text(self, text: str) -> str:
        """
        Clean raw extracted text by removing noise and normalizing format.

        Args:
            text (str): Raw text to clean.

        Returns:
            str: Cleaned text.
        """
        if not text:
            return ""

        try:
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text.strip())

            # Remove headers/footers
            # for pattern in self.header_footer_patterns:
            #     text = re.sub(pattern, "", text, flags=re.IGNORECASE)

            # Remove special characters (preserve diacritics)
            text = re.sub(self.diacritics, "", text)

            # Normalize multiple spaces again
            text = re.sub(r"\s+", " ", text.strip())

            return text
        except Exception as e:
            self.logger.error("Text cleaning failed: %s", str(e))
            return ""

    def is_valid_text(self, text: str) -> bool:
        """
        Validate if cleaned text is meaningful.

        Args:
            text (str): Cleaned text to validate.

        Returns:
            bool: True if text is valid (sufficient length and content).
        """
        if not text or len(text.strip()) < self.min_text_length:
            return False

        # Check for Italian diacritics or legal terms
        if re.search(r'[àèìòùÀÈÌÒÙ]', text, re.IGNORECASE): #or re.search(r'\b(legge|decreto|articolo)\b', text, re.IGNORECASE):
            return True

        # Fallback: At least 5 words
        words = text.split()
        return len(words) >= 5

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single text file, clean it, and return metadata.

        Args:
            file_path (Path): Path to the input text file.

        Returns:
            Dict[str, Any]: Metadata about cleaning result, without cleaned text.
        """
        result = {
            "file_path": Path(self.output_dir, file_path.name).as_posix(),
            "file_name": file_path.name,
            "is_valid": False,
            "error": None,
            "original_length": 0,
            "cleaned_length": 0,
            "fingerprint": ""
        }

        self.logger.info("Cleaning text file: %s", file_path)
        try:
            # Read the text file
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            result["original_length"] = len(raw_text)

            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            result["cleaned_length"] = len(cleaned_text)
            result["is_valid"] = self.is_valid_text(cleaned_text)

            # Generate text fingerprint for deduplication
            result["fingerprint"] = fingerprint(cleaned_text)

            if not result["is_valid"]:
                result["error"] = "Cleaned text seems invalid (too short or lacks meaningful content)"
                self.logger.warning(result["error"])

            # Save cleaned text only (no metadata file)
            self.save_cleaned_text(file_path.name, cleaned_text)

            return result

        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_path, str(e))
            result["error"] = str(e)
            return result

    def save_cleaned_text(self, file_name: str, cleaned_text: str) -> None:
        """
        Save cleaned text to output directory.

        Args:
            file_name (str): Name of the original file.
            cleaned_text (str): Cleaned text content to save.
        """
        file_stem = file_name.rsplit(".", 1)[0]
        text_file = self.output_dir / f"{file_stem}.txt"

        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            self.logger.info("Saved cleaned text to %s", text_file)
        except Exception as e:
            self.logger.error("Failed to save cleaned text to %s: %s", text_file, str(e))

    def process_directory(self) -> None:
        """
        Process all text files in the input directory and save aggregated metadata.
        """
        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning("No text files found in %s", self.input_dir)
            return

        self.logger.info("Processing %d text files in %s", len(text_files), self.input_dir)
        processed_files = 0
        metadata_collection = []

        for file_path in text_files:
            metadata = self.process_file(file_path)
            metadata_collection.append(metadata)
            processed_files += 1

        self.logger.info("Processed %d/%d text files", processed_files, len(text_files))

        # Save all metadata in a single summary JSON file (without cleaned_text field)
        summary_file = self.output_dir / "cleaning_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(metadata_collection, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved cleaning summary metadata to %s", summary_file)
        except Exception as e:
            self.logger.error("Failed to save cleaning summary metadata: %s", str(e))


if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        cleaner = TextCleaner(
            input_dir='data/prefettura_v1.3_texts',
            output_dir='data/prefettura_v1.3.1_cleaned_texts',
            min_text_length=20
        )
        cleaner.process_directory()
        print("Text cleaning completed.")
    except Exception as e:
        print(f"Error during text cleaning: {e}")
