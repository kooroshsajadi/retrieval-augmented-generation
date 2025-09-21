import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import logging
from src.ingestion.ingest_data import DataIngestor

class TestDataIngestor(unittest.TestCase):
    def setUp(self):
        # Set up logger and DataIngestor instance
        self.logger = logging.getLogger("data_ingestor")
        self.output_dir = "data/texts"
        self.ingestor = DataIngestor(
            output_dir=self.output_dir,
            max_pages=None,
            language="ita",
            tessdata_dir=None,
            logger=self.logger
        )

    @patch("scripts.ingest_data.pdfplumber.open")
    def test_extract_text_from_pdf_text_based(self, mock_pdfplumber):
        # Arrange
        file_path = "data/source/sample.pdf"
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value = mock_pdf
        expected_text = "Sample PDF text\n"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.pdf")
        self.assertEqual(result["file_type"], "pdf")
        self.assertEqual(result["text"], expected_text)
        self.assertTrue(result["is_valid"])
        self.assertIsNone(result["error"])
        mock_pdfplumber.assert_called_once_with(Path(file_path))

    @patch("scripts.ingest_data.pdfplumber.open")
    @patch("scripts.ingest_data.convert_from_path")
    @patch("scripts.ingest_data.pytesseract.image_to_string")
    @patch("scripts.ingest_data.Image.open")
    def test_extract_text_from_pdf_ocr(self, mock_image_open, mock_pytesseract, mock_convert, mock_pdfplumber):
        # Arrange
        file_path = "data/source/sample.pdf"
        mock_pdfplumber.side_effect = Exception("No text")  # Simulate text-based extraction failure
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_image_enhanced = MagicMock()
        mock_image.enhance.return_value = mock_image_enhanced
        mock_convert.return_value = [mock_image]
        mock_pytesseract.return_value = "Sample OCR text"
        expected_text = "Sample OCR text\n"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.pdf")
        self.assertEqual(result["file_type"], "pdf")
        self.assertEqual(result["text"], expected_text)
        self.assertTrue(result["is_valid"])
        self.assertIsNone(result["error"])
        mock_convert.assert_called_once_with(Path(file_path))
        mock_pytesseract.assert_called_once_with(mock_image_enhanced, lang="ita")

    @patch("builtins.open", new_callable=mock_open, read_data="Sample text file content")
    def test_extract_text_from_txt(self, mock_file):
        # Arrange
        file_path = "data/source/sample.txt"
        expected_text = "Sample text file content"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.txt")
        self.assertEqual(result["file_type"], "txt")
        self.assertEqual(result["text"], expected_text)
        self.assertTrue(result["is_valid"])
        self.assertIsNone(result["error"])
        mock_file.assert_called_once_with(Path(file_path), "r", encoding="utf-8")

    @patch("scripts.ingest_data.Image.open")
    @patch("scripts.ingest_data.pytesseract.image_to_string")
    def test_extract_text_from_image(self, mock_pytesseract, mock_image_open):
        # Arrange
        file_path = "data/source/sample.png"
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_image_enhanced = MagicMock()
        mock_image.enhance.return_value = mock_image_enhanced
        mock_pytesseract.return_value = "Sample image text"
        expected_text = "Sample image text"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.png")
        self.assertEqual(result["file_type"], "png")
        self.assertEqual(result["text"], expected_text)
        self.assertTrue(result["is_valid"])
        self.assertIsNone(result["error"])
        mock_image_open.assert_called_once_with(Path(file_path))
        mock_pytesseract.assert_called_once_with(mock_image_enhanced, lang="ita")

    @patch("scripts.ingest_data.Path")
    def test_extract_text_unsupported_file_type(self, mock_path):
        # Arrange
        file_path = "data/source/sample.doc"
        mock_path_instance = MagicMock()
        mock_path_instance.name = "sample.doc"
        mock_path_instance.suffix.lower.return_value = ".doc"
        mock_path.return_value = mock_path_instance

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["file_name"], "sample.doc")
        self.assertEqual(result["file_type"], "doc")
        self.assertEqual(result["text"], "")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error"], "Unsupported file type: doc")

    @patch("builtins.open", new_callable=mock_open)
    @patch("scripts.ingest_data.pdfplumber.open")
    def test_save_extracted_text(self, mock_pdfplumber, mock_file):
        # Arrange
        file_path = "data/source/sample.pdf"
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value = mock_pdf
        expected_text = "Sample PDF text\n"

        # Act
        result = self.ingestor.extract_text(file_path)

        # Assert
        self.assertTrue(result["is_valid"])
        mock_file.assert_called_once_with(Path(self.output_dir) / "sample.txt", "w", encoding="utf-8")
        mock_file().write.assert_called_once_with(expected_text)

if __name__ == "__main__":
    unittest.main()