import logging
from typing import List, Dict, Any
from pymilvus import Collection
from sentence_transformers import SentenceTransformer
from src.utils.logging_utils import setup_logger
import torch

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
        self.collection_name = collection_name

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model,
                device="xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {embedding_model}: {str(e)}")
            raise

        # Connect to Milvus
        try:
            from pymilvus import connections
            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            self.collection = Collection(collection_name)
            self.collection.load()
            self.logger.info(f"Connected to Milvus collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def encode_query(self, query: str) -> List[float]:
        """Encode query into embedding vector."""
        try:
            embedding = self.embedding_model.encode(
                [query],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embedding.cpu().numpy()[0].tolist()
        except Exception as e:
            self.logger.error(f"Failed to encode query '{query}': {str(e)}")
            raise

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
            query_vector = self.encode_query(query)
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
            self.logger.info(f"Retrieved {len(retrieved)} chunks for query: {query[:50]}...")
            return retrieved
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            raise