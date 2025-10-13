import logging
from typing import List, Dict, Any, Optional
from src.retrieval.milvus_connector import MilvusConnector
from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.models.cross_encoders import CrossEncoderModels
from src.utils.models.bi_encoders import EncoderModels
from src.utils.logging_utils import setup_logger

class MilvusRetriever:
    """Orchestrator for hybrid retrieval and reranking in RAG pipeline."""
    def __init__(
        self,
        collection_name: str = "gotmat_collection",
        embedding_model: str = EncoderModels.ITALIAN_LEGAL_BERT_SC.value,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        reranker_model: str = "dlicari/Italian-Legal-BERT",  # Italian-specific cross-encoder
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MilvusRetriever with hybrid retrieval and reranking capabilities.

        Args:
            collection_name (str): Milvus collection name.
            embedding_model (str): SentenceTransformer model for query encoding.
            milvus_host (str): Milvus server host.
            milvus_port (str): Milvus server port.
            reranker_model (str): Cross-encoder model for reranking.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger("src.retrieval.MilvusRetriever")
        self.encoder = QueryEncoder(embedding_model=embedding_model, logger=self.logger)
        self.connector = MilvusConnector(
            collection_name=collection_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            logger=self.logger
        )
        self.hybrid_retriever = HybridRetriever(
            milvus_connector=self.connector,
            query_encoder=self.encoder,
            collection_name=collection_name,
            logger=self.logger
        )
        self.reranker = Reranker(
            model_name=reranker_model,
            logger=self.logger
        )

    def retrieve(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank top-k relevant chunks using hybrid retrieval.

        Args:
            query (str): User query (in Italian).
            top_k (int): Number of chunks to retrieve after reranking.
            hybrid_weight (float): Weight for vector search score in hybrid retrieval (0 to 1).

        Returns:
            List[Dict[str, Any]]: Reranked child chunks with chunk_id, text, score, parent_id, parent_file_path.
        """
        try:
            # Step 1: Hybrid retrieval
            hybrid_results = self.hybrid_retriever.retrieve_hybrid(
                query=query,
                top_k=top_k * 3,
                vector_weight=hybrid_weight
            )
            self.logger.info(f"Hybrid retrieval returned {len(hybrid_results)} chunks for query: {query[:50]}...")

            # Step 2: Rerank results
            reranked_results = self.reranker.rerank(
                query=query,
                chunks=hybrid_results,
                top_k=top_k
            )
            self.logger.info(f"Reranked to {len(reranked_results)} chunks for query: {query[:50]}...")
            return reranked_results

        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            raise

    def fetch_parent_texts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch parent chunk texts for the retrieved child chunks.

        Args:
            results (List[Dict[str, Any]]): Retrieved chunks with parent_file_path.

        Returns:
            List[Dict[str, Any]]: Results with added parent_text field.
        """
        try:
            for result in results:
                parent_file_path = result.get("parent_file_path")
                if parent_file_path:
                    try:
                        with open(parent_file_path, "r", encoding="utf-8") as f:
                            result["parent_text"] = f.read()
                        self.logger.debug(f"Loaded parent text from {parent_file_path}")
                    except FileNotFoundError:
                        self.logger.warning(f"Parent file {parent_file_path} not found")
                        result["parent_text"] = ""
                else:
                    result["parent_text"] = ""
            return results
        except Exception as e:
            self.logger.error(f"Failed to fetch parent texts: {str(e)}")
            raise