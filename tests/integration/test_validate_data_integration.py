import unittest
from pathlib import Path
import logging
from src.validation.validate_data import DataValidator

class TestDataValidatorIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("data_validator")
        self.validator = DataValidator(logger=self.logger)
        self.test_files = {
            "pdf": "data/test/116876.pdf",
            "txt": "data/test/BodyPart.txt",
            "png": "data/test/1000017202.jpg",
            "invalid": "data/test/invalid.zip",
            # "nonexistent": "data/test/nonexistent.pdf"
        }

    def test_validate_pdf(self):
        file_path = self.test_files["pdf"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")
        result = self.validator.validate_file(file_path)
        self.assertTrue(result["is_valid"], f"Validation failed: {result['error']}")
        self.assertIsNone(result["error"])

    def test_validate_txt(self):
        file_path = self.test_files["txt"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")
        result = self.validator.validate_file(file_path)
        self.assertTrue(result["is_valid"], f"Validation failed: {result['error']}")
        self.assertIsNone(result["error"])

    def test_validate_png(self):
        file_path = self.test_files["png"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")
        result = self.validator.validate_file(file_path)
        self.assertTrue(result["is_valid"], f"Validation failed: {result['error']}")
        self.assertIsNone(result["error"])

def test_validate_unsupported_format(self):
        file_path = self.test_files["invalid"]
        extension = Path(file_path).suffix.lower()
        result = self.validator.validate_file(file_path)
        self.assertFalse(result["is_valid"])
        self.assertEqual(
            result["error"],
            f"Unsupported file format: {extension}. Supported formats: {', '.join(self.validator.supported_formats)}"
        )
        print(f"Attached file format: {extension}")

    # def test_validate_nonexistent_file(self):
    #     file_path = self.test_files["nonexistent"]
    #     result = self.validator.validate_file(file_path)
    #     self.assertFalse(result["is_valid"])
    #     self.assertEqual(result["error"], f"File not found: {Path(file_path)}")

if __name__ == "__main__":
    unittest.main()