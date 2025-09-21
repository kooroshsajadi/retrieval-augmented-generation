from typing import Union
import torch
from transformers import AutoModel
from utils.models.abstract_model_loader import AbstractModelLoader

class EncoderOnlyModelLoader(AbstractModelLoader):
    """Model loader for encoder-only models."""

    model_type = "encoder-only"
    model_class = AutoModel
    expected_configs = tuple(AutoModel._model_mapping.keys())

    def generate(self, text: str, max_new_tokens: int = 50) -> Union[str, torch.Tensor]:
        """
        Generate embeddings for encoder-only models (mean-pooled hidden states).

        Args:
            text (str): Input text.
            max_new_tokens (int): Ignored for encoder-only models.

        Returns:
            torch.Tensor: Embeddings.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Return mean-pooled embeddings (mean of hidden states)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate: {str(e)}")
            raise