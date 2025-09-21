import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from src.utils.logging_utils import setup_logger

class Reranker:
    """Reranks retrieved chunks using a cross-encoder model."""
    def __init__(self, model_name: str = "dlicari/Italian-Legal-BERT", logger: Optional[logging.Logger] = None):
        """
        Initialize Reranker.

        Args:
            model_name (str): Cross-encoder model name (Italian-specific).
            logger (logging.Logger): Logger instance.
        """
        self.model = CrossEncoder(model_name, max_length=512)
        self.logger = logger or setup_logger("src.retrieval.reranker")
        self.logger.info(f"Loaded cross-encoder model: {model_name}")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on cross-encoder scores.

        Args:
            query (str): User query (in Italian).
            chunks (List[Dict[str, Any]]): Retrieved chunks with chunk_id, text, score, parent_id, parent_file_path.
            top_k (int): Number of chunks to return after reranking.

        Returns:
            List[Dict[str, Any]]: Reranked chunks with chunk_id, text, score, parent_id, parent_file_path.
        """
        try:
            # Prepare query-chunk pairs
            pairs = [[query, chunk["text"]] for chunk in chunks]
            scores = self.model.predict(pairs)

            # Update scores and preserve metadata
            reranked_chunks = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                    "parent_id": chunk.get("parent_id"),
                    "parent_file_path": chunk.get("parent_file_path")
                }
                for chunk, score in zip(chunks, scores)
            ]
            reranked_chunks = sorted(reranked_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
            return reranked_chunks

        except Exception as e:
            self.logger.error(f"Reranking failed for query '{query}': {str(e)}")
            raise