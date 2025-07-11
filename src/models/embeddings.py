from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel(ABC):
    """Abstract base class for embedding models, defining the interface for text embedding generation."""
    
    @abstractmethod
    def encode(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    """Implements text embedding generation using a SentenceTransformer model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str | list[str]) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)