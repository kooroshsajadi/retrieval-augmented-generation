from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

# TODO: Adapt the main retrieval to this interface.

class Retriever(ABC):
    """Abstract base class for retrieval models, defining the interface for document retrieval."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents with their scores."""
        pass

class VectorStoreRetriever(Retriever):
    """Implements document retrieval using a vector store and query embeddings."""
    
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