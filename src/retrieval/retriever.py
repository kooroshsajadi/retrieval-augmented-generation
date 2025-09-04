import logging
from typing import List, Dict, Any
from src.retrieval.milvus_connector import MilvusConnector
from src.retrieval.query_encoder import QueryEncoder
from src.utils.logging_utils import setup_logger

class MilvusRetriever:
    """Retriever for querying Milvus vector database in RAG pipeline."""

    def __init__(
        self,
        collection_name: str = "legal_texts",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        logger: logging.Logger = None
    ):
        """
        Initialize MilvusRetriever.

        Args:
            collection_name (str): Milvus collection name.
            embedding_model (str): SentenceTransformer model for query encoding.
            milvus_host (str): Milvus server host.
            milvus_port (str): Milvus server port.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)
        self.encoder = QueryEncoder(embedding_model=embedding_model, logger=self.logger)
        self.connector = MilvusConnector(
            collection_name=collection_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            logger=self.logger
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks from Milvus.

        Args:
            query (str): User query (in Italian).
            top_k (int): Number of chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with chunk_id, text, and distance.
        """
        try:
            query_vector = self.encoder.encode_query(query)
            results = self.connector.search(query_vector, top_k)
            self.logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            raise