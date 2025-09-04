import unittest
from pathlib import Path
import numpy as np
import json
import logging
from scripts.sentence_transformer import SentenceTransformerEmbedder

class TestSentenceTransformerIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("sentence_transformer_test")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.output_dir = "data/test/embeddings"
        self.embedder = SentenceTransformerEmbedder(
            model_name="intfloat/multilingual-e5-large",
            output_dir=self.output_dir,
            max_chunk_words=500,
            min_chunk_length=10,
            logger=self.logger
        )
        self.test_files = {
            "pdf": "data/test/116876.pdf",
            "txt": "data/test/BodyPart.txt",
            "jpg": "data/test/1000017202.jpg"
        }
        self.test_query = "Quali sono i requisiti per la residenza in Italia?"
        # Ensure output directory is clean
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        summary_file = Path(self.output_dir) / "embeddings_summary.json"
        if summary_file.exists():
            summary_file.unlink()

    def test_process_query(self):
        # Arrange
        query = self.test_query

        # Act
        result = self.embedder.process_query(query)

        # Assert
        self.assertTrue(result["is_valid"], f"Query embedding failed: {result['error']}")
        self.assertIsNone(result["error"])
        self.assertEqual(result["query"], query)
        self.assertIsInstance(result["embedding"], np.ndarray)
        self.assertGreater(result["embedding"].size, 0, "Embedding is empty")
        self.logger.info("Query embedding test passed")

    def test_process_file_pdf(self):
        # Arrange
        file_path = self.test_files["pdf"]
        text_file = Path("data/test/texts") / f"{Path(file_path).stem}.txt"
        if not Path(file_path).exists() or not text_file.exists():
            self.skipTest(f"Test file or extracted text not found: {file_path}, {text_file}")
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()

        # Act
        result = self.embedder.process_file(file_path, extracted_text)

        # Assert
        self.assertTrue(result["is_valid"], f"File embedding failed: {result['error']}")
        self.assertIsNone(result["error"])
        self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertGreater(len(result["chunk_embeddings"]), 0, "No chunks generated")
        for chunk in result["chunk_embeddings"]:
            self.assertTrue(chunk["is_valid"])
            self.assertIsInstance(chunk["text"], str)
            self.assertGreater(len(chunk["text"].strip()), 0, "Chunk text is empty")
            embedding_file = Path(self.output_dir) / chunk["embedding_file"]
            self.assertTrue(embedding_file.exists(), f"Embedding file not found: {embedding_file}")
            embedding = np.load(embedding_file)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertGreater(embedding.size, 0, f"Embedding empty for {chunk['chunk_id']}")
        # Check summary file
        summary_file = Path(self.output_dir) / "embeddings_summary.json"
        self.assertTrue(summary_file.exists(), "Embeddings summary not found")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertGreater(len(summary), 0, "Embeddings summary is empty")
        self.logger.info("PDF file embedding test passed")

    def test_process_file_txt(self):
        # Arrange
        file_path = self.test_files["txt"]
        text_file = Path("data/test/texts") / f"{Path(file_path).stem}.txt"
        if not Path(file_path).exists() or not text_file.exists():
            self.skipTest(f"Test file or extracted text not found: {file_path}, {text_file}")
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()

        # Act
        result = self.embedder.process_file(file_path, extracted_text)

        # Assert
        self.assertTrue(result["is_valid"], f"File embedding failed: {result['error']}")
        self.assertIsNone(result["error"])
        self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertGreater(len(result["chunk_embeddings"]), 0, "No chunks generated")
        for chunk in result["chunk_embeddings"]:
            self.assertTrue(chunk["is_valid"])
            self.assertIsInstance(chunk["text"], str)
            self.assertGreater(len(chunk["text"].strip()), 0, "Chunk text is empty")
            embedding_file = Path(self.output_dir) / chunk["embedding_file"]
            self.assertTrue(embedding_file.exists(), f"Embedding file not found: {embedding_file}")
            embedding = np.load(embedding_file)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertGreater(embedding.size, 0, f"Embedding empty for {chunk['chunk_id']}")
        # Check summary file
        summary_file = Path(self.output_dir) / "embeddings_summary.json"
        self.assertTrue(summary_file.exists(), "Embeddings summary not found")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertGreater(len(summary), 0, "Embeddings summary is empty")
        self.logger.info("Text file embedding test passed")

    def test_process_file_image(self):
        # Arrange
        file_path = self.test_files["jpg"]
        text_file = Path("data/test/texts") / f"{Path(file_path).stem}.txt"
        if not Path(file_path).exists() or not text_file.exists():
            self.skipTest(f"Test file or extracted text not found: {file_path}, {text_file}")
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()

        # Act
        result = self.embedder.process_file(file_path, extracted_text)

        # Assert
        self.assertTrue(result["is_valid"], f"File embedding failed: {result['error']}")
        self.assertIsNone(result["error"])
        self.assertEqual(result["file_path"], Path(file_path).as_posix())
        self.assertGreater(len(result["chunk_embeddings"]), 0, "No chunks generated")
        for chunk in result["chunk_embeddings"]:
            self.assertTrue(chunk["is_valid"])
            self.assertIsInstance(chunk["text"], str)
            self.assertGreater(len(chunk["text"].strip()), 0, "Chunk text is empty")
            embedding_file = Path(self.output_dir) / chunk["embedding_file"]
            self.assertTrue(embedding_file.exists(), f"Embedding file not found: {embedding_file}")
            embedding = np.load(embedding_file)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertGreater(embedding.size, 0, f"Embedding empty for {chunk['chunk_id']}")
        # Check summary file
        summary_file = Path(self.output_dir) / "embeddings_summary.json"
        self.assertTrue(summary_file.exists(), "Embeddings summary not found")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertGreater(len(summary), 0, "Embeddings summary is empty")
        self.logger.info("Image file embedding test passed")

    def test_process_file_missing_text(self):
        # Arrange
        file_path = "data/test/missing.pdf"
        text_file = Path("data/test/texts") / f"{Path(file_path).stem}.txt"

        # Act
        result = self.embedder.process_file(file_path)

        # Assert
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error"], f"Extracted text file not found: {text_file}")
        self.assertEqual(len(result["chunk_embeddings"]), 0)
        self.logger.info("Missing text file test passed")

if __name__ == "__main__":
    unittest.main()