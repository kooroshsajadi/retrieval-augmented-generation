import unittest
from pathlib import Path
import logging
from scripts.ingest_data import DataIngestor

class TestDataIngestorIntegration(unittest.TestCase):
    def setUp(self):
        # Set up logger and DataIngestor instance
        self.logger = logging.getLogger("data_ingestor")
        self.output_dir = "data/texts"
        self.ingestor = DataIngestor(
            output_dir=self.output_dir,
            max_pages=None,
            language="ita",
            tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata",  # Adjust to your Tesseract path
            logger=self.logger
        )
        # Sample file paths for testing (replace with actual paths to your test files)
        self.test_files = {
            "pdf": "data/source/sample.pdf",
            "txt": "data/source/sample.txt",
            "png": "data/source/sample.png"
        }

    def test_extract_text_from_pdf(self):
        # Arrange
        file_path = self.test_files["pdf"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.pdf")
        self.assertEqual(result["file_type"], "pdf")
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / "sample.txt"
        self.assertTrue(output_file.exists(), f"Output file not created: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            saved_text = f.read()
        self.assertEqual(saved_text, result["text"])

    def test_extract_text_from_txt(self):
        # Arrange
        file_path = self.test_files["txt"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.txt")
        self.assertEqual(result["file_type"], "txt")
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / "sample.txt"
        self.assertTrue(output_file.exists(), f"Output file not created: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            saved_text = f.read()
        self.assertEqual(saved_text, result["text"])

    def test_extract_text_from_image(self):
        # Arrange
        file_path = self.test_files["png"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.png")
        self.assertEqual(result["file_type"], "png")
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / "sample.txt"
        self.assertTrue(output_file.exists(), f"Output file not created: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            saved_text = f.read()
        self.assertEqual(saved_text, result["text"])

    def test_extract_text_unsupported_file_type(self):
        # Arrange
        file_path = "data/source/sample.doc"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.doc")
        self.assertEqual(result["file_type"], "doc")
        self.assertEqual(result["text"], "")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error"], "Unsupported file type: doc")
        # Check that no output file is created
        output_file = Path(self.output_dir) / "sample.txt"
        self.assertFalse(output_file.exists(), f"Output file should not exist: {output_file}")

if __name__ == "__main__":
    unittest.main()