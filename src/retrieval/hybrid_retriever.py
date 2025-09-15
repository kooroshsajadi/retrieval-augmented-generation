import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.retrieval.milvus_connector import MilvusConnector
from src.retrieval.query_encoder import QueryEncoder

class HybridRetriever:
    """Handles hybrid retrieval combining vector and keyword-based search."""
    def __init__(
        self,
        milvus_connector: MilvusConnector,
        query_encoder: QueryEncoder,
        collection_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HybridRetriever.

        Args:
            milvus_connector (MilvusConnector): Milvus connector instance.
            query_encoder (QueryEncoder): Query encoder instance.
            collection_name (str): Milvus collection name.
            logger: Optional[logging.Logger]: Optional logger instance.
        """
        self.milvus_connector = milvus_connector
        self.query_encoder = query_encoder
        self.collection_name = collection_name
        self.logger = logger or logging.getLogger("src.retrieval.hybrid_retriever")
        self.bm25 = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        try:
            texts, ids = self.milvus_connector.get_all_texts()
            tokenized_texts = [text.split() for text in texts]
            self.bm25 = BM25Okapi(tokenized_texts)
            self.texts = texts
            self.ids = ids
            self.logger.info("BM25 index initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 index: {str(e)}")
            raise

    def retrieve_hybrid(self, query: str, top_k: int, vector_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining vector and BM25 search.

        Args:
            query (str): User query (in Italian).
            top_k (int): Number of chunks to retrieve.
            vector_weight (float): Weight for vector search score (0 to 1).

        Returns:
            List[Dict[str, Any]]: Merged results with chunk_id, text, and score.
        """
        try:
            # Vector search
            query_vector = self.query_encoder.encode_query(query)
            vector_results = self.milvus_connector.search(query_vector, top_k)
            vector_scores = {res["chunk_id"]: res["distance"] for res in vector_results}

            # BM25 search
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results = [
                {"chunk_id": self.ids[i], "text": self.texts[i], "score": score}
                for i, score in enumerate(bm25_scores) if score > 0
            ]
            bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:top_k]

            # Normalize scores (vector: cosine similarity, BM25: raw scores)
            max_vector_score = max(vector_scores.values(), default=1.0)
            max_bm25_score = max([r["score"] for r in bm25_results], default=1.0) or 1.0
            merged_results = {}
            for res in vector_results:
                chunk_id = res["chunk_id"]
                score = vector_scores[chunk_id] / max_vector_score
                merged_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": res["text"],
                    "score": score * vector_weight
                }
            for res in bm25_results:
                chunk_id = res["chunk_id"]
                score = res["score"] / max_bm25_score
                if chunk_id in merged_results:
                    merged_results[chunk_id]["score"] += score * (1 - vector_weight)
                else:
                    merged_results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": res["text"],
                        "score": score * (1 - vector_weight)
                    }

            # Sort by combined score
            final_results = list(merged_results.values())
            final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
            return final_results

        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed for query '{query}': {str(e)}")
            raise