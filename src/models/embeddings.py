from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModels(Enum):
    # Embedding size: 1024, suitable for high-performance machines (GPU with 8GB+ VRAM)
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    
    # Embedding size: 1024, suitable for high-performance machines (GPU with 8GB+ VRAM)
    MULTILINGUAL_E5_LARGE_INSTRUCT = "intfloat/multilingual-e5-large-instruct"
    
    # Embedding size: 768 (after pooling), suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM)
    BERT_BASE_ITALIAN_CASED = "dbmdz/bert-base-italian-cased"
    
    # Embedding size: 768, suitable for medium-performance machines (CPU or GPU with 4GB+ VRAM)
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2 = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Embedding size: 512, suitable for low-performance machines (CPU or GPU with 2GB+ VRAM)
    DISTILUSE_BASE_MULTILINGUAL_CASED_V2 = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    
    # Embedding size: 4096, suitable for very high-performance machines (GPU with 24GB+ VRAM or multi-GPU)
    E5_MISTRAL_7B_INSTRUCT = "intfloat/e5-mistral-7b-instruct"
    
    # Embedding size: 1024 (after pooling), suitable for high-performance machines (GPU with 12GB+ VRAM)
    XLM_ROBERTA_LARGE = "xlm-roberta-large"
    
    # Cross-encoder for reranking, outputs scalar score (not embeddings), suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM)
    MS_MARCO_MINILM_L12_V2 = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Cross-encoder for reranking, outputs scalar score (not embeddings), optimized for Italian, suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM)
    MMARCO_MINILM_L12_H384_V1 = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

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