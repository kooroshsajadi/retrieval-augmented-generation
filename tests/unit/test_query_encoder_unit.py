import unittest
from unittest.mock import patch, MagicMock
import logging
from src.retrieval.query_encoder import QueryEncoder
import numpy as np

class TestQueryEncoder(unittest.TestCase):
    def setUp(self):
        # Set up a logger for testing
        self.logger = logging.getLogger(__name__)
        self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        self.query = "Test query"

    @patch("src.retrieval.query_encoder.SentenceTransformer")
    @patch("src.retrieval.query_encoder.torch")
    def test_init_success(self, mock_torch, mock_sentence_transformer):
        # Arrange
        mock_torch.xpu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_model_instance

        # Act
        encoder = QueryEncoder(
            embedding_model=self.embedding_model,
            logger=self.logger
        )

        # Assert
        mock_sentence_transformer.assert_called_once_with(
            self.embedding_model, device="cpu"
        )
        self.assertEqual(encoder.embedding_model, mock_model_instance)

    @patch("src.retrieval.query_encoder.SentenceTransformer")
    @patch("src.retrieval.query_encoder.torch")
    def test_init_model_load_failure(self, mock_torch, mock_sentence_transformer):
        # Arrange
        mock_torch.xpu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_sentence_transformer.side_effect = Exception("Model load failed")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            QueryEncoder(
                embedding_model=self.embedding_model,
                logger=self.logger
            )
        self.assertEqual(str(context.exception), "Model load failed")

    @patch("src.retrieval.query_encoder.SentenceTransformer")
    @patch("src.retrieval.query_encoder.torch")
    def test_encode_query_success(self, mock_torch, mock_sentence_transformer):
        # Arrange
        mock_torch.xpu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_model_instance
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model_instance.encode.return_value = mock_tensor

        encoder = QueryEncoder(
            embedding_model=self.embedding_model,
            logger=self.logger
        )

        # Act
        embedding = encoder.encode_query(self.query)

        # Assert
        mock_model_instance.encode.assert_called_once_with(
            [self.query], convert_to_tensor=True, show_progress_bar=False
        )
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch("src.retrieval.query_encoder.SentenceTransformer")
    @patch("src.retrieval.query_encoder.torch")
    def test_encode_query_failure(self, mock_torch, mock_sentence_transformer):
        # Arrange
        mock_torch.xpu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_model_instance
        mock_model_instance.encode.side_effect = Exception("Encoding failed")

        encoder = QueryEncoder(
            embedding_model=self.embedding_model,
            logger=self.logger
        )

        # Act & Assert
        with self.assertRaises(Exception) as context:
            encoder.encode_query(self.query)
        self.assertEqual(str(context.exception), "Encoding failed")

if __name__ == "__main__":
    unittest.main()