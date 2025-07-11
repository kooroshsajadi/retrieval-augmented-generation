from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from src.models.embeddings import EmbeddingModel

class Retriever(ABC):
    @abstractmethod # Abstracts retrieval to support different strategies (e.g., BM25, hybrid search)
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents with their scores."""
        pass

class VectorStoreRetriever(Retriever):
    def __init__(self, embedding_model: EmbeddingModel, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store  # Assume data layer provides vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Query vector store (assumes vector_store has a search method)
        results = self.vector_store.search(query_embedding, top_k)
        
        # Return list of (document, score) tuples
        return [(doc["text"], doc["score"]) for doc in results]