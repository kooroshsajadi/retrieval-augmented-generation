import unittest
import logging
import csv
import shutil
from pathlib import Path
import numpy as np
from pymilvus import connections, has_collection, Collection, DataType
from src.data.vector_store import VectorStore
from src.utils.logging_utils import setup_logger

class TestVectorStoreIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment, logger, and temporary directories."""
        self.logger = setup_logger("src.data.test_vector_store_integration")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Test parameters
        self.collection_name = "test_gotmat_collection"
        self.milvus_host = "localhost"
        self.milvus_port = "19530"
        self.embedding_dim = 1024
        self.chunks_dir = Path("data/test/chunks")
        self.embeddings_dir = Path("data/test/embeddings")
        self.results_dir = Path("data/test/results")
        self.texts_dir = Path("data/test/texts")

        # Create directories
        for dir_path in [self.chunks_dir, self.embeddings_dir, self.results_dir, self.texts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # CSV file for results
        self.csv_file = self.results_dir / "vector_store_test_results.csv"
        self.csv_data = []

        # Verify Milvus connectivity
        try:
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            self.logger.info("Connected to Milvus")
        except Exception as e:
            self.logger.error("Failed to connect to Milvus: %s", str(e))
            self.skipTest(f"Failed to connect to Milvus: {str(e)}")

        # Clean up any existing test collection
        if has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.logger.info("Dropped existing test collection: %s", self.collection_name)

        # Initialize VectorStore
        try:
            self.vector_store = VectorStore(
                collection_name=self.collection_name,
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port,
                embedding_dim=self.embedding_dim,
                chunks_dir=self.chunks_dir,
                embeddings_dir=self.embeddings_dir,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error("Failed to initialize VectorStore: %s", str(e))
            self.fail(f"Failed to initialize VectorStore: {str(e)}")

    def _save_csv(self):
        """Save test results to CSV."""
        try:
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Test Case", "Status", "Details"])
                writer.writerows(self.csv_data)
            self.logger.info("Saved test results to %s", self.csv_file)
        except Exception as e:
            self.logger.error("Failed to save CSV file %s: %s", self.csv_file, str(e))
            self.fail(f"Failed to save CSV file: {str(e)}")

    def _create_sample_data(self, num_chunks: int = 3):
        """Create sample chunk texts and embeddings."""
        texts = [f"Sample text {i}" for i in range(num_chunks)]
        embeddings = [np.random.rand(self.embedding_dim).astype(np.float32) for _ in range(num_chunks)]
        chunk_ids = [f"test_chunk_{i}" for i in range(num_chunks)]

        # Write chunk files
        for chunk_id, text in zip(chunk_ids, texts):
            with open(self.chunks_dir / f"{chunk_id}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            np.save(self.embeddings_dir / f"{chunk_id}.npy", embeddings[chunk_ids.index(chunk_id)])

        return texts, embeddings, chunk_ids

    def test_collection_creation(self):
        """Test collection creation and schema."""
        test_case = "Collection Creation"
        try:
            # Verify collection exists
            self.assertTrue(has_collection(self.collection_name), "Collection not created")
            collection = Collection(self.collection_name)
            
            # Check schema
            schema = collection.schema
            expected_fields = {
                "id": DataType.INT64,
                "embedding": DataType.FLOAT_VECTOR,
                "chunk_id": DataType.VARCHAR,
                "text": DataType.VARCHAR,
                "subject": DataType.VARCHAR
            }
            for field in schema.fields:
                self.assertIn(field.name, expected_fields, f"Unexpected field {field.name}")
                self.assertEqual(field.dtype, expected_fields[field.name], f"Field {field.name} has incorrect type")
                if field.name == "embedding":
                    self.assertEqual(field.params["dim"], self.embedding_dim, "Embedding dimension mismatch")
            
            # Check number of entities (should be 0 initially)
            self.assertEqual(collection.num_entities, 0, "Collection should be empty")
            
            self.csv_data.append([test_case, "PASS", f"Collection {self.collection_name} created with correct schema"])
            self.logger.info("Collection creation test passed")
        except Exception as e:
            self.csv_data.append([test_case, "FAIL", f"Error: {str(e)}"])
            self.logger.error("Collection creation test failed: %s", str(e))
            self.fail(f"Collection creation test failed: {str(e)}")

    def test_bulk_insert(self):
        """Test bulk insertion of chunks and embeddings."""
        test_case = "Bulk Insert"
        try:
            # Create sample data
            texts, embeddings, chunk_ids = self._create_sample_data(num_chunks=3)

            # Perform bulk insert
            success = self.vector_store.bulk_insert()
            self.assertTrue(success, "Bulk insert failed")

            # Verify collection state
            collection = Collection(self.collection_name)
            collection.load()
            self.assertEqual(collection.num_entities, len(texts), f"Expected {len(texts)} entities, got {collection.num_entities}")

            # Verify data integrity
            results = collection.query(expr="id >= 0", output_fields=["id", "chunk_id", "text", "subject"])
            self.assertEqual(len(results), len(texts), "Query result count mismatch")
            for i, result in enumerate(results):
                self.assertEqual(result["chunk_id"], chunk_ids[i], f"Chunk ID mismatch for id {i}")
                self.assertEqual(result["text"], texts[i], f"Text mismatch for id {i}")
                self.assertEqual(result["subject"], "courthouse", f"Subject mismatch for id {i}")

            # Verify embedding dimension
            embedding_results = collection.query(expr="id >= 0", output_fields=["embedding"])
            for result in embedding_results:
                self.assertEqual(len(result["embedding"]), self.embedding_dim, "Embedding dimension mismatch")

            # Verify index
            self.assertTrue(collection.has_index(), "Index not created")
            index = collection.index()
            self.assertEqual(index.params["index_type"], "IVF_FLAT", "Incorrect index type")
            self.assertEqual(index.params["metric_type"], "L2", "Incorrect metric type")

            self.csv_data.append([test_case, "PASS", f"Inserted {len(texts)} entities, verified schema, index, and data"])
            self.logger.info("Bulk insert test passed")
        except Exception as e:
            self.csv_data.append([test_case, "FAIL", f"Error: {str(e)}"])
            self.logger.error("Bulk insert test failed: %s", str(e))
            self.fail(f"Bulk insert test failed: {str(e)}")

    def test_incremental_insert(self):
        """Test incremental insertion of a single chunk."""
        test_case = "Incremental Insert"
        try:
            # Create sample data
            texts, embeddings, chunk_ids = self._create_sample_data(num_chunks=1)

            # Perform incremental insert
            success = self.vector_store.store_vectors(texts, embeddings, chunk_ids, subject="courthouse", force_recreate=False)
            self.assertTrue(success, "Incremental insert failed")

            # Verify collection state
            collection = Collection(self.collection_name)
            collection.load()
            self.assertEqual(collection.num_entities, 1, "Expected 1 entity after incremental insert")

            # Verify data integrity
            results = collection.query(expr="id >= 0", output_fields=["id", "chunk_id", "text", "subject"])
            self.assertEqual(len(results), 1, "Query result count mismatch")
            self.assertEqual(results[0]["chunk_id"], chunk_ids[0], "Chunk ID mismatch")
            self.assertEqual(results[0]["text"], texts[0], "Text mismatch")
            self.assertEqual(results[0]["subject"], "courthouse", "Subject mismatch")

            # Verify embedding dimension
            embedding_results = collection.query(expr="id >= 0", output_fields=["embedding"])
            self.assertEqual(len(embedding_results[0]["embedding"]), self.embedding_dim, "Embedding dimension mismatch")

            self.csv_data.append([test_case, "PASS", "Incremental insert successful, verified data"])
            self.logger.info("Incremental insert test passed")
        except Exception as e:
            self.csv_data.append([test_case, "FAIL", f"Error: {str(e)}"])
            self.logger.error("Incremental insert test failed: %s", str(e))
            self.fail(f"Incremental insert test failed: {str(e)}")

    def test_invalid_embedding_dimension(self):
        """Test insertion with invalid embedding dimension."""
        test_case = "Invalid Embedding Dimension"
        try:
            texts = ["Sample text"]
            embeddings = [np.random.rand(512).astype(np.float32)]  # Wrong dimension
            chunk_ids = ["invalid_chunk"]
            success = self.vector_store.store_vectors(texts, embeddings, chunk_ids)
            self.assertFalse(success, "Insert should fail with invalid embedding dimension")
            self.csv_data.append([test_case, "PASS", "Correctly rejected invalid embedding dimension"])
            self.logger.info("Invalid embedding dimension test passed")
        except Exception as e:
            self.csv_data.append([test_case, "FAIL", f"Error: {str(e)}"])
            self.logger.error("Invalid embedding dimension test failed: %s", str(e))
            self.fail(f"Invalid embedding dimension test failed: {str(e)}")

    def tearDown(self):
        """Clean up test collection and save CSV results."""
        try:
            if has_collection(self.collection_name):
                Collection(self.collection_name).drop()
                self.logger.info("Dropped test collection: %s", self.collection_name)
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
            shutil.rmtree(self.embeddings_dir, ignore_errors=True)
            self._save_csv()
        except Exception as e:
            self.logger.error("Teardown failed: %s", str(e))
            self.fail(f"Teardown failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()