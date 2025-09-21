from typing import Union
import torch
from transformers import AutoModelForSeq2SeqLM
from utils.models.abstract_model_loader import AbstractModelLoader

class Seq2SeqModelLoader(AbstractModelLoader):
    """Model loader for seq2seq models."""

    model_type = "seq2seq"
    model_class = AutoModelForSeq2SeqLM
    expected_configs = tuple(AutoModelForSeq2SeqLM._model_mapping.keys())

    def generate(self, text: str, max_new_tokens: int = 50) -> Union[str, torch.Tensor]:
        """
        Generate text for seq2seq models (e.g., translation or summary).

        Args:
            text (str): Input text.
            max_new_tokens (int): Maximum new tokens for generation.

        Returns:
            str: Generated text.
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
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate: {str(e)}")
            raise