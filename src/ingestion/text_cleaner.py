import json
import re
from typing import Dict, List, Any
from pathlib import Path
import logging

import yaml
from src.utils.logging_utils import setup_logger

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
            output_dir (str): Directory to save cleaned text and metadata.
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
            r"Page\s+\d+\s*(of\s+\d+)?",  # e.g., "Page 1", "Page 1 of 5"
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # e.g., "12/31/2023"
            r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?",  # e.g., "12:30", "12:30:45 PM"
        ]
        self.special_char_pattern = r"[^\w\sàèìòùÀÈÌÒÙ]"  # Keep Italian diacritics

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
            for pattern in self.header_footer_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)

            # Remove special characters (preserve diacritics)
            text = re.sub(self.special_char_pattern, "", text)

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
        if re.search(r'[àèìòù]', text, re.IGNORECASE) or re.search(r'\b(legge|decreto|articolo)\b', text, re.IGNORECASE):
            return True

        # Fallback: At least 5 words
        words = text.split()
        return len(words) >= 5

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single text file, clean it, and save results.

        Args:
            file_path (Path): Path to the input text file.

        Returns:
            Dict[str, Any]: Cleaning result with text and metadata.
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "cleaned_text": "",
            "is_valid": False,
            "error": None,
            "original_length": 0,
            "cleaned_length": 0
        }

        self.logger.info("Cleaning text file: %s", file_path)
        try:
            # Read text file
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            result["original_length"] = len(raw_text)

            # Clean text
            cleaned_text = self.clean_text(raw_text)
            result["cleaned_text"] = cleaned_text
            result["cleaned_length"] = len(cleaned_text)
            result["is_valid"] = self.is_valid_text(cleaned_text)

            if not result["is_valid"]:
                result["error"] = "Cleaned text is invalid (too short or lacks meaningful content)"
                self.logger.warning(result["error"])

            # Save results
            self.save_cleaned_text(result)
            return result

        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_path, str(e))
            result["error"] = str(e)
            return result

    def save_cleaned_text(self, result: Dict[str, Any]) -> None:
        """
        Save cleaned text and metadata to output directory.

        Args:
            result (Dict[str, Any]): Cleaning result with text and metadata.
        """
        file_name = result["file_name"].rsplit(".", 1)[0]
        text_file = self.output_dir / f"{file_name}.txt"
        metadata_file = self.output_dir / f"{file_name}_metadata.json"

        # Save cleaned text
        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(result["cleaned_text"])
            self.logger.info("Saved cleaned text to %s", text_file)
        except Exception as e:
            self.logger.error("Failed to save cleaned text to %s: %s", text_file, str(e))
            result["error"] = str(e)

        # Save metadata
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved cleaning metadata to %s", metadata_file)
        except Exception as e:
            self.logger.error("Failed to save metadata to %s: %s", metadata_file, str(e))
            result["error"] = str(e)

    def process_directory(self) -> None:
        """
        Process all text files in the input directory.
        """
        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning("No text files found in %s", self.input_dir)
            return

        self.logger.info("Processing %d text files in %s", len(text_files), self.input_dir)
        processed_files = 0
        results = []

        for file_path in text_files:
            result = self.process_file(file_path)
            results.append(result)
            processed_files += 1

        self.logger.info("Processed %d/%d text files", processed_files, len(text_files))

        # Save summary metadata
        summary_file = self.output_dir / "cleaning_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved cleaning summary to %s", summary_file)
        except Exception as e:
            self.logger.error("Failed to save cleaning summary: %s", str(e))

    def get_cleaning_results(self) -> List[Dict[str, Any]]:
        """
        Load cleaning results from metadata files in output directory.

        Returns:
            List[Dict[str, Any]]: List of cleaning result dictionaries.
        """
        summary_file = self.output_dir / "cleaning_summary.json"
        if not summary_file.exists():
            self.logger.warning("Cleaning summary file not found: %s", summary_file)
            return []

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load cleaning summary: %s", str(e))
            return []

if __name__ == "__main__":
    with open('src/data/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        cleaner = TextCleaner(
            input_dir=config['texts']['prefettura_v1'],
            output_dir=config['cleaned_texts']['prefettura_v1'],
            min_text_length=20
        )
        cleaner.process_directory()
        print("Text cleaning completed.")
    except Exception as e:
        print(f"Error during text cleaning: {e}")