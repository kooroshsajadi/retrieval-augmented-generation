from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel(ABC):
    @abstractmethod # Abstracts embedding logic to support different models (e.g., OpenAI embeddings, custom models)
    def encode(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        pass

class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str | list[str]) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)