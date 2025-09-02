import logging
from typing import List, Dict, Any
import torch
from src.models.model_loader import ModelLoader
from src.utils.logging_utils import setup_logger
from typing import Optional

class LLMGenerator:
    """Generator using fine-tuned seq2seq model for response generation in RAG pipeline."""

    def __init__(
        self,
        model_path: str = "models/fine_tuned_models/opus-mt-it-en",
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        model_type: str = "seq2seq",
        max_length: int = 128,
        device: str = "auto",
        logger: logging.Logger = None
    ):
        """
        Initialize LLMGenerator.

        Args:
            model_path (str): Path to fine-tuned model.
            adapter_path: Optional[str]: Path to optional adapters.
            tokenizer_path: Optional[str]: Path to optional tokenizer.
            model_type (str): Model type ("seq2seq").
            max_length (int): Maximum sequence length.
            device (str): Device ("auto", "cpu", "xpu").
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)
        self.max_length = max_length

        # Load model using ModelLoader
        try:
            self.model_loader = ModelLoader(
                model_name=model_path,
                adapter_path=adapter_path,
                tokenizer_path=tokenizer_path,
                model_type=model_type,
                device_map=device,
                max_length=max_length,
                logger=self.logger
            )
            self.model = self.model_loader.model
            self.tokenizer = self.model_loader.tokenizer
            self.device = self.model_loader.device
            self.logger.info(f"Loaded model {model_path} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelLoader: {str(e)}")
            raise

    def generate(self, query: str, contexts: List[Dict[str, Any]], max_new_tokens: int = 50) -> str:
        """
        Generate response in Italian using query and retrieved contexts.

        Args:
            query (str): User query (in Italian).
            contexts (List[Dict[str, Any]]): Retrieved chunks with 'text' field.
            max_new_tokens (int): Maximum tokens to generate.

        Returns:
            str: Generated response in Italian.
        """
        try:
            # Create prompt
            context_texts = [ctx["text"] for ctx in contexts]
            prompt = f"Domanda: {query}\nContesto: {' '.join(context_texts)}\nRisposta:"
            self.logger.info(f"Generated prompt: {prompt[:100]}...")

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info(f"Generated response: {response[:100]}...")
            return response
        except Exception as e:
            self.logger.error(f"Generation failed for query '{query}': {str(e)}")
            raise