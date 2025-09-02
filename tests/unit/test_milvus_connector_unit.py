import unittest
from unittest.mock import patch, MagicMock
import logging
from src.retrieval.milvus_connector import MilvusConnector

class TestMilvusConnector(unittest.TestCase):
    def setUp(self):
        # Set up a logger for testing
        self.logger = logging.getLogger(__name__)
        self.collection_name = "test_collection"
        self.milvus_host = "localhost"
        self.milvus_port = "19530"

    @patch("src.retrieval.milvus_connector.connections")
    @patch("src.retrieval.milvus_connector.Collection")
    def test_init_success(self, mock_collection, mock_connections):
        # Arrange
        mock_collection_instance = MagicMock()
        mock_collection.return_value = mock_collection_instance

        # Act
        connector = MilvusConnector(
            collection_name=self.collection_name,
            milvus_host=self.milvus_host,
            milvus_port=self.milvus_port,
            logger=self.logger
        )

        # Assert
        mock_connections.connect.assert_called_once_with(
            alias="default", host=self.milvus_host, port=self.milvus_port
        )
        mock_collection.assert_called_once_with(self.collection_name)
        mock_collection_instance.load.assert_called_once()
        self.assertEqual(connector.collection_name, self.collection_name)
        self.assertEqual(connector.collection, mock_collection_instance)

    @patch("src.retrieval.milvus_connector.connections")
    def test_init_connection_failure(self, mock_connections):
        # Arrange
        mock_connections.connect.side_effect = Exception("Connection failed")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            MilvusConnector(
                collection_name=self.collection_name,
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port,
                logger=self.logger
            )
        self.assertEqual(str(context.exception), "Connection failed")

    @patch("src.retrieval.milvus_connector.connections")
    @patch("src.retrieval.milvus_connector.Collection")
    def test_search_success(self, mock_collection, mock_connections):
        # Arrange
        mock_collection_instance = MagicMock()
        mock_collection.return_value = mock_collection_instance
        query_vector = [0.1, 0.2, 0.3]
        top_k = 2
        mock_hit1 = MagicMock()
        mock_hit1.entity.get.side_effect = lambda key: {"chunk_id": "1", "text": "Test text 1"}[key]
        mock_hit1.distance = 0.5
        mock_hit2 = MagicMock()
        mock_hit2.entity.get.side_effect = lambda key: {"chunk_id": "2", "text": "Test text 2"}[key]
        mock_hit2.distance = 0.6
        mock_results = [[mock_hit1, mock_hit2]]
        mock_collection_instance.search.return_value = mock_results

        connector = MilvusConnector(
            collection_name=self.collection_name,
            milvus_host=self.milvus_host,
            milvus_port=self.milvus_port,
            logger=self.logger
        )

        # Act
        results = connector.search(query_vector, top_k)

        # Assert
        mock_collection_instance.search.assert_called_once_with(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["chunk_id", "text"]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "1")
        self.assertEqual(results[0]["text"], "Test text 1")
        self.assertEqual(results[0]["distance"], 0.5)
        self.assertEqual(results[1]["chunk_id"], "2")
        self.assertEqual(results[1]["text"], "Test text 2")
        self.assertEqual(results[1]["distance"], 0.6)

    @patch("src.retrieval.milvus_connector.connections")
    @patch("src.retrieval.milvus_connector.Collection")
    def test_search_failure(self, mock_collection, mock_connections):
        # Arrange
        mock_collection_instance = MagicMock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.search.side_effect = Exception("Search failed")
        query_vector = [0.1, 0.2, 0.3]
        top_k = 2

        connector = MilvusConnector(
            collection_name=self.collection_name,
            milvus_host=self.milvus_host,
            milvus_port=self.milvus_port,
            logger=self.logger
        )

        # Act & Assert
        with self.assertRaises(Exception) as context:
            connector.search(query_vector, top_k)
        self.assertEqual(str(context.exception), "Search failed")

if __name__ == "__main__":
    unittest.main()