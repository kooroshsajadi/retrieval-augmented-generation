import unittest
import logging
import json
import shutil
from pathlib import Path
import numpy as np
from pymilvus import connections, has_collection, Collection
from src.utils.logging_utils import setup_logger
from main import RAGOrchestrator
import yaml

class TestRAGOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment and logger."""
        self.logger = setup_logger("tests.integration.test_rag_orchestrator_integration")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Test parameters
        self.config_path = "configs/test_rag.yaml"
        self.collection_name = "test_gotmat_collection"
        self.test_dir = Path("data/test")
        self.output_dir = self.test_dir / "results"
        self.chunks_dir = "data/chunks"
        self.embeddings_dir = "data/embeddings"
        self.texts_dir = self.test_dir / "texts"
        self.queries_file = self.test_dir / "prompts.json"
        self.output_file = self.output_dir / "responses.json"

        # Test files
        self.test_files = [
            self.test_dir / "files/116876.pdf",
            self.test_dir / "files/BodyPart.txt",
            self.test_dir / "files/1000017202.jpg"
        ]

        # Create test config
        config = {
            "data": {
                "texts": self.texts_dir.as_posix(),
                "chunks": self.chunks_dir.as_posix(),
                "embeddings": self.embeddings_dir.as_posix(),
                "destination": self.output_dir.as_posix()
            },
            "supported_formats": [".txt", ".pdf", ".jpg"],
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_dim": 1024,
            "collection_name": self.collection_name,
            "milvus_host": "localhost",
            "milvus_port": "19530",
            "tessdata_dir": r"C:\Program Files\Tesseract-OCR\tessdata",
            "max_chunk_words": 500,
            "min_chunk_length": 10,
            "max_contexts": 3,
            "max_context_length": 1000,
            "model_path": "models/fine_tuned_models/opus-mt-it-en",
            "model_type": "seq2seq",
            "max_length": 128,
            "max_new_tokens": 50,
            "device": "cpu"  # Use CPU for testing to avoid GPU issues
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        # Create test directories
        for dir_path in [self.output_dir, self.chunks_dir, self.embeddings_dir, self.texts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create sample queries file
        sample_queries = [
            {"prompt": 1, "Italian": "Quali sono le conseguenze del pagamento di una sanzione amministrativa in Italia?", "English": "What are the consequences of paying an administrative fine in Italy?"},
            {"prompt": 2, "Italian": "Entro quanti giorni posso presentare scritti difensivi dopo la notifica del verbale?", "English": "Within how many days can I submit defense writings after the notification of the report?"}
        ]
        with open(self.queries_file, "w", encoding="utf-8") as f:
            json.dump(sample_queries, f, ensure_ascii=False, indent=2)

        # Verify Milvus connectivity
        try:
            connections.connect(alias="default", host="localhost", port="19530")
            self.logger.info("Connected to Milvus")
        except Exception as e:
            self.logger.error("Failed to connect to Milvus: %s", str(e))
            self.skipTest(f"Failed to connect to Milvus: {str(e)}")

        # Clean up existing collection
        if has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.logger.info("Dropped existing test collection: %s", self.collection_name)

        # Initialize RAGOrchestrator
        self.orchestrator = RAGOrchestrator(config_path=self.config_path)

    def test_process_file_and_queries(self):
        """Test file processing and query handling with JSON output."""
        test_case = "Process File and Queries"
        try:
            # Process test files
            for file_path in self.test_files:
                if not file_path.exists():
                    self.logger.warning("Test file not found: %s, skipping", file_path)
                    continue
                success = self.orchestrator.process_file(file_path.as_posix())
                self.assertTrue(success, f"Failed to process file: {file_path}")

            # Verify Milvus collection
            self.assertTrue(has_collection(self.collection_name), "Collection not created")
            collection = Collection(self.collection_name)
            collection.load()
            self.assertGreater(collection.num_entities, 0, "No entities inserted into collection")

            # Process queries from file
            success = self.orchestrator.process_queries_from_file(self.queries_file.as_posix(), self.output_file.as_posix())
            self.assertTrue(success, "Failed to process queries from file")

            # Verify output JSON
            self.assertTrue(self.output_file.exists(), f"Output file not created: {self.output_file}")
            with open(self.output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # Check JSON structure
            expected_queries = [
                "Quali sono le conseguenze del pagamento di una sanzione amministrativa in Italia?",
                "Entro quanti giorni posso presentare scritti difensivi dopo la notifica del verbale?"
            ]
            self.assertEqual(len(results), len(expected_queries), "Incorrect number of responses")
            for result, expected_query in zip(results, expected_queries):
                self.assertIn("query", result, "Missing 'query' field in result")
                self.assertIn("answer", result, "Missing 'answer' field in result")
                self.assertEqual(len(result), 2, "Result contains unexpected fields")
                self.assertEqual(result["query"], expected_query, "Query mismatch")
                self.assertIsInstance(result["answer"], str, "Answer is not a string")
                self.assertFalse(result["answer"].startswith("Error:"), f"Answer contains error: {result['answer']}")

            self.logger.info("File and query processing test passed")
        except Exception as e:
            self.logger.error("File and query processing test failed: %s", str(e))
            self.fail(f"File and query processing test failed: {str(e)}")

    def test_retrieval_details(self):
        """Test retrieval details including chunk_id, text, and distance."""
        test_case = "Retrieval Details"
        try:
            # Process a test file to populate Milvus
            file_path = self.test_dir / "files/116876.pdf"
            if not file_path.exists():
                self.skipTest(f"Test file not found: {file_path}")

            success = self.orchestrator.process_file(file_path.as_posix())
            self.assertTrue(success, f"Failed to process file: {file_path}")

            # Process a single query and check contexts
            query = "Quali sono le conseguenze del pagamento di una sanzione amministrativa in Italia?"
            result = self.orchestrator.process_query(query, top_k=3)
            self.assertIn("query", result, "Missing 'query' field")
            self.assertIn("response", result, "Missing 'response' field")
            self.assertIn("contexts", result, "Missing 'contexts' field")
            self.assertEqual(result["query"], query, "Query mismatch")
            self.assertIsInstance(result["response"], str, "Response is not a string")
            self.assertFalse(result["response"].startswith("Error:"), f"Response contains error: {result['response']}")

            # Verify context details
            contexts = result["contexts"]
            self.assertGreater(len(contexts), 0, "No contexts retrieved")
            for context in contexts:
                self.assertIn("chunk_id", context, "Missing 'chunk_id' in context")
                self.assertIn("text", context, "Missing 'text' in context")
                self.assertIn("distance", context, "Missing 'distance' in context")
                self.assertIsInstance(context["chunk_id"], str, "Chunk ID is not a string")
                self.assertIsInstance(context["text"], str, "Text is not a string")
                self.assertIsInstance(context["distance"], float, "Distance is not a float")
                self.assertGreater(len(context["text"]), 0, "Context text is empty")

            self.logger.info("Retrieval details test passed")
        except Exception as e:
            self.logger.error("Retrieval details test failed: %s", str(e))
            self.fail(f"Retrieval details test failed: {str(e)}")

    def test_invalid_queries_file(self):
        """Test handling of invalid queries file."""
        test_case = "Invalid Queries File"
        try:
            invalid_file = self.test_dir / "nonexistent.json"
            success = self.orchestrator.process_queries_from_file(invalid_file.as_posix(), self.output_file.as_posix())
            self.assertFalse(success, "Processing should fail for nonexistent queries file")
            self.assertFalse(self.output_file.exists(), "Output file should not be created")
            self.logger.info("Invalid queries file test passed")
        except Exception as e:
            self.logger.error("Invalid queries file test failed: %s", str(e))
            self.fail(f"Invalid queries file test failed: {str(e)}")

    def tearDown(self):
        """Clean up test environment."""
        try:
            if has_collection(self.collection_name):
                Collection(self.collection_name).drop()
                self.logger.info("Dropped test collection: %s", self.collection_name)
            for dir_path in [self.output_dir, self.chunks_dir, self.embeddings_dir, self.texts_dir]:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    self.logger.info("Cleaned up directory: %s", dir_path)
            if Path(self.config_path).exists():
                Path(self.config_path).unlink()
                self.logger.info("Removed test config: %s", self.config_path)
            if self.queries_file.exists():
                self.queries_file.unlink()
                self.logger.info("Removed test queries file: %s", self.queries_file)
        except Exception as e:
            self.logger.error("Teardown failed: %s", str(e))
            self.fail(f"Teardown failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()