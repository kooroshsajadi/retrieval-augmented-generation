import subprocess
from typing import Union
from src.utils.models.base_model_loader import BaseModelLoaderInterface
from src.utils.logging_utils import setup_logger  # Import your logger if you want to use it

class OllamaModelLoader(BaseModelLoaderInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = setup_logger("ollama_model_loader")  # Optional: Add logging

    def generate(self, text: str, max_new_tokens: int = 50, repetition_penalty: float = 1.0, temperature: float = 0.7, top_p: float = 0.9) -> str:
        # Log warning about ignored params (since CLI doesn't support them; use Modelfile instead)
        self.logger.warning(
            "Dynamic parameters (max_new_tokens=%d, repetition_penalty=%.2f, temperature=%.2f, top_p=%.2f) are ignored for Ollama CLI. "
            "Use Modelfile settings for customization.",
            max_new_tokens, repetition_penalty, temperature, top_p
        )

        # Pipe the prompt into 'ollama run <model>' for non-interactive generation
        cmd = ["ollama", "run", self.model_name, text]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            output = result.stdout.strip()
            if result.stderr:
                self.logger.error("Ollama stderr: %s", result.stderr)
            return output
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama generation failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Ollama subprocess failed: {str(e)}. Ensure 'ollama' is installed and the model is pulled.")