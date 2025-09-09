import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.utils.logging_utils import setup_logger
import torch

class QueryEncoder:
    """Handles query encoding using SentenceTransformer model."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QueryEncoder.

        Args:
            embedding_model (str): SentenceTransformer model for query encoding.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)

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

    def encode_query(self, query: str) -> List[float]:
        """
        Encode query into embedding vector.

        Args:
            query (str): Query text to encode.

        Returns:
            List[float]: Encoded query vector.
        """
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