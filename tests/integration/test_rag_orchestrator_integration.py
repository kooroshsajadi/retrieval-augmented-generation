import unittest
import logging
import csv
from pathlib import Path
from scripts.main import RAGOrchestrator
from src.utils.logging_utils import setup_logger

class TestRAGOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        """Set up the test environment and logger."""
        self.logger = setup_logger("tests.integration.test_rag_orchestrator_integration")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Paths
        self.config_path = "configs/rag.yaml"
        self.test_files = {
            "pdf": "data/test/116876.pdf",
            "txt": "data/test/BodyPart.txt",
            "jpg": "data/test/1000017202.jpg"
        }
        self.test_query = "Quali sono i requisiti per la residenza in Italia?"
        self.results_dir = Path("data/test/results")
        self.texts_dir = Path("data/test/texts")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # CSV file for results
        self.csv_file = self.results_dir / "test_results.csv"
        self.csv_data = []

        # Initialize RAGOrchestrator with real dependencies
        try:
            self.orchestrator = RAGOrchestrator(config_path=self.config_path)
        except Exception as e:
            self.logger.error("Failed to initialize RAGOrchestrator: %s", str(e))
            self.fail(f"Setup failed: {str(e)}")

    def _save_csv(self):
        """Save test results to CSV in data/test/results."""
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Input", "Response"])  # Header
            writer.writerows(self.csv_data)
        self.logger.info("Saved test results to %s", self.csv_file)

    def test_process_query_only(self):
        """Test query processing without a file input."""
        response = self.orchestrator.process_query(self.test_query, top_k=2)
        self.assertIsInstance(response, str)
        self.assertNotIn("Error:", response, f"Query processing failed: {response}")
        self.assertGreater(len(response), 0, "Response is empty")
        self.csv_data.append([self.test_query, response])
        self.logger.info("Query-only test passed: %s", response[:50])

    # def test_process_file_and_query_pdf(self):
    #     """Test file processing and query with a PDF file."""
    #     file_path = self.test_files["pdf"]
    #     text_file = self.texts_dir / f"{Path(file_path).stem}.txt"
        
    #     # Skip if test file or extracted text doesn't exist
    #     if not Path(file_path).exists():
    #         self.skipTest(f"Test file not found: {file_path}")
    #     if not text_file.exists():
    #         self.skipTest(f"Extracted text not found: {text_file}")

    #     # Process file
    #     success = self.orchestrator.process_file(file_path)
    #     self.assertTrue(success, f"File processing failed for {file_path}")

    #     # Process query
    #     response = self.orchestrator.process_query(self.test_query, top_k=2)
    #     self.assertIsInstance(response, str)
    #     self.assertNotIn("Error:", response, f"Query processing failed: {response}")
    #     self.assertGreater(len(response), 0, "Response is empty")
    #     self.csv_data.append([text_file.as_posix(), response])
    #     self.logger.info("PDF file and query test passed: %s", response[:50])

    # def test_process_file_and_query_txt(self):
    #     """Test file processing and query with a text file."""
    #     file_path = self.test_files["txt"]
    #     text_file = self.texts_dir / f"{Path(file_path).stem}.txt"
        
    #     # Skip if test file or extracted text doesn't exist
    #     if not Path(file_path).exists():
    #         self.skipTest(f"Test file not found: {file_path}")
    #     if not text_file.exists():
    #         self.skipTest(f"Extracted text not found: {text_file}")

    #     # Process file
    #     success = self.orchestrator.process_file(file_path)
    #     self.assertTrue(success, f"File processing failed for {file_path}")

    #     # Process query
    #     response = self.orchestrator.process_query(self.test_query, top_k=2)
    #     self.assertIsInstance(response, str)
    #     self.assertNotIn("Error:", response, f"Query processing failed: {response}")
    #     self.assertGreater(len(response), 0, "Response is empty")
    #     self.csv_data.append([text_file.as_posix(), response])
    #     self.logger.info("Text file and query test passed: %s", response[:50])

    # def test_process_file_and_query_image(self):
    #     """Test file processing and query with an image file."""
    #     file_path = self.test_files["jpg"]
    #     text_file = self.texts_dir / f"{Path(file_path).stem}.txt"
        
    #     # Skip if test file or extracted text doesn't exist
    #     if not Path(file_path).exists():
    #         self.skipTest(f"Test file not found: {file_path}")
    #     if not text_file.exists():
    #         self.skipTest(f"Extracted text not found: {text_file}")

    #     # Process file
    #     success = self.orchestrator.process_file(file_path)
    #     self.assertTrue(success, f"File processing failed for {file_path}")

    #     # Process query
    #     response = self.orchestrator.process_query(self.test_query, top_k=2)
    #     self.assertIsInstance(response, str)
    #     self.assertNotIn("Error:", response, f"Query processing failed: {response}")
    #     self.assertGreater(len(response), 0, "Response is empty")
    #     self.csv_data.append([text_file.as_posix(), response])
    #     self.logger.info("Image file and query test passed: %s", response[:50])

    # def test_process_invalid_file(self):
    #     """Test processing an invalid file."""
    #     file_path = "data/test/missing.pdf"
    #     success = self.orchestrator.process_file(file_path)
    #     self.assertFalse(success, "Processing should fail for invalid file")
    #     self.csv_data.append([file_path, "Error: File not found"])
    #     self.logger.info("Invalid file test passed")

    # def test_process_invalid_query(self):
    #     """Test processing an invalid (empty) query."""
    #     response = self.orchestrator.process_query("")
    #     self.assertIn("Error:", response, "Empty query should return an error")
    #     self.csv_data.append(["", response])
    #     self.logger.info("Invalid query test passed")

    # def test_csv_output(self):
    #     """Verify CSV file is created with correct content."""
    #     # Run a query to generate data
    #     self.orchestrator.process_query(self.test_query)
    #     self._save_csv()
    #     with open(self.csv_file, "r", encoding="utf-8") as f:
    #         reader = csv.reader(f)
    #         headers = next(reader)
    #         self.assertEqual(headers, ["Input", "Response"], "CSV headers incorrect")
    #         rows = list(reader)
    #         self.assertGreater(len(rows), 0, "CSV is empty")
    #         self.assertIn(self.test_query, [row[0] for row in rows], "Query not found in CSV")
    #     self.logger.info("CSV output test passed")

    def tearDown(self):
        """Save CSV results after all tests."""
        self._save_csv()

if __name__ == "__main__":
    unittest.main()