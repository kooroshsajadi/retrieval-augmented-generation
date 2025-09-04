import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.logging_utils import setup_logger

class LLMGenerator:
    """Generates responses using a language model for the RAG pipeline."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        model_type: str = "seq2seq",
        max_length: int = 128,
        device: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLMGenerator with model and tokenizer.

        Args:
            model_path (str): Path or name of the language model.
            adapter_path (Optional[str]): Path to model adapter, if any.
            tokenizer_path (Optional[str]): Path to tokenizer, if different from model.
            model_type (str): Type of model (e.g., 'seq2seq').
            max_length (int): Maximum input length for tokenization.
            device (str): Device to run model on ('auto', 'cpu', 'cuda').
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.logger = logger or setup_logger("llm_generator")
        self.max_length = max_length
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            if adapter_path:
                self.model.load_adapter(adapter_path)
            self.model.to(self.device)
            self.logger.info("Loaded model %s on %s", model_path, self.device)
        except Exception as e:
            self.logger.error("Failed to load model or tokenizer: %s", str(e))
            raise

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate a response from a formatted prompt.

        Args:
            prompt (str): Input prompt containing query and contexts.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Generated response.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info("Generated response for prompt: %s...", prompt[:50])
            return response
        except Exception as e:
            self.logger.error("Generation failed: %s", str(e))
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    generator = LLMGenerator(
        model_path="facebook/opus-mt-it-en",
        max_length=128,
        device="auto"
    )
    prompt = """Query: Quali sono i requisiti per la residenza in Italia?
Context:
1. (courthouse) Per la residenza in Italia, è necessario un passaporto valido...
2. (courthouse) I requisiti includono un'assicurazione sanitaria valida..."""
    response = generator.generate(prompt, max_new_tokens=50)
    print(f"Response: {response}")