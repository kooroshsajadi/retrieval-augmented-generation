import logging
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection
from src.utils.logging_utils import setup_logger

class MilvusConnector:
    """Handles connection and search operations for Milvus vector database."""

    def __init__(
        self,
        collection_name: str = "legal_texts",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MilvusConnector.

        Args:
            collection_name (str): Milvus collection name.
            milvus_host (str): Milvus server host.
            milvus_port (str): Milvus server port.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)
        self.collection_name = collection_name

        # Connect to Milvus
        try:
            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            self.collection = Collection(collection_name)
            self.collection.load()
            self.logger.info(f"Connected to Milvus collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector search in Milvus collection.

        Args:
            query_vector (List[float]): Query embedding vector.
            top_k (int): Number of chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with chunk_id, text, and distance.
        """
        try:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_id", "text"]
            )
            retrieved = [
                {
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text": hit.entity.get("text"),
                    "distance": hit.distance
                }
                for hit in results[0]
            ]
            self.logger.info(f"Retrieved {len(retrieved)} chunks")
            return retrieved
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise