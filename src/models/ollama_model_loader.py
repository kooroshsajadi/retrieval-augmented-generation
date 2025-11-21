from typing import Union
import subprocess
import torch
from transformers import AutoModelForCausalLM
from src.utils.models.abstract_model_loader import AbstractModelLoader


class OllamaModelLoader(AbstractModelLoader):
    """Model loader for Ollama models using GGUF locally."""

    model_type = "ollama"

    def __init__(self, model_name: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device

    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using Ollama.

        Args:
            text (str): Input prompt text.
            max_new_tokens (int): Max tokens to generate.

        Returns:
            str: Generated text.
        """
        try:
            # Ollama CLI command example: ollama generate <model_name> --prompt "<text>" --max-tokens <max_new_tokens>
            cmd = [
                "ollama", "generate", self.model_name,
                "--prompt", text,
                "--max-tokens", str(max_new_tokens)
            ]
            # Running the subprocess and capturing output
            completed_process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            generated_text = completed_process.stdout.strip()
            return generated_text
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ollama generation failed: {e.stderr}")
            raise
