from typing import Union
import torch
from transformers import AutoModelForCausalLM
from src.utils.models.abstract_model_loader import AbstractModelLoader

class CausalModelLoader(AbstractModelLoader):
    """Model loader for causal (decoder-only) models."""

    model_type = "causal"
    model_class = AutoModelForCausalLM
    expected_configs = tuple(AutoModelForCausalLM._model_mapping.keys())

    def generate(self, text: str, max_new_tokens: int = 50) -> Union[str, torch.Tensor]:
        """
        Generate text for causal models (e.g., text completion).

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