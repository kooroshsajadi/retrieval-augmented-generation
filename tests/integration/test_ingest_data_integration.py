import unittest
from pathlib import Path
import logging
from scripts.ingest_data import DataIngestor

class TestDataIngestorIntegration(unittest.TestCase):
    def setUp(self):
        # Set up logger and DataIngestor instance
        self.logger = logging.getLogger("tests.integration.test_ingest_data_integration")
        self.output_dir = "data/test/extracted_texts"
        self.ingestor = DataIngestor(
            output_dir=self.output_dir,
            max_pages=None,
            language="ita",
            tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata",  # Adjust to your Tesseract path
            logger=self.logger
        )
        # Sample file paths for testing
        self.test_files = {
            "pdf": "data/test/files/FACCOEzequiel03021998Brasile-trasmisisoneverbaleviolazioneArt688CPdel23022022-PFPadova.pdf",
            "pdf": "data/test/files/FACCOEzequiel03021998BrasiletrasmissioneOrdinanazaPrefetturaPadovanotificataPFPadova.pdf",
            "pdf": "data/test/files/FACCOEzequiel03021998BrasiletrasmissioneverbaleviolazioneamministrativaPFPadova.pdf",
            "pdf": "data/test/files/FarellaGraziellaRaysnc.pdf",
            "pdf": "data/test/files/116876.pdf",
            "txt": "data/test/files/BodyPart.txt",
            "png": "data/test/files/1000017202.jpg"
        }

    def test_extract_text_from_pdf(self):
        # Arrange
        file_path = self.test_files["pdf"]
        if not Path(file_path).exists():
            self.skipTest(f"Test file not found: {file_path}")

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertGreater(len(result["text"].strip()), 0, "Extracted text is empty")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / f"{Path(file_path).stem}.txt"
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
        # self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertGreater(len(result["text"].strip()), 0, "Extracted text is empty")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / f"{Path(file_path).stem}.txt"
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
        # self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["is_valid"], f"Extraction failed: {result['error']}")
        self.assertGreater(len(result["text"].strip()), 0, "Extracted text is empty")
        self.assertIsNone(result["error"])
        # Check if text file was saved
        output_file = Path(self.output_dir) / f"{Path(file_path).stem}.txt"
        self.assertTrue(output_file.exists(), f"Output file not created: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            saved_text = f.read()
        self.assertEqual(saved_text, result["text"])

    # def test_extract_text_unsupported_file_type(self):
    #     # Arrange
    #     file_path = "data/test/files/invalid.zip"

    #     # Act
    #     result = self.ingestor.extract_text(file_path)

    #     # Assert
    #     # self.assertEqual(result["file_path"], Path(file_path).as_posix())
    #     self.assertEqual(result["text"], "")
    #     self.assertFalse(result["is_valid"])
    #     self.assertEqual(result["error"], "Unsupported file type: doc")
    #     # Check that no output file is created
    #     output_file = Path(self.output_dir) / "sample.txt"
    #     self.assertFalse(output_file.exists(), f"Output file should not exist: {output_file}")

if __name__ == "__main__":
    unittest.main()