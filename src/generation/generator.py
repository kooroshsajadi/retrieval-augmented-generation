import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.model_loader import ModelLoader
from src.utils.logging_utils import setup_logger
import torch
from src.utils.models.model_utils import create_and_configure_tokenizer

class LLMGenerator:
    """Generates responses using a language model for the RAG pipeline."""

    def __init__(
        self,
        model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        model_type: str = "causal",
        max_length: int = 2048,
        device: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLMGenerator with model and tokenizer.

        Args:
            model_path (str): Path or name of the language model (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct').
            adapter_path (Optional[str]): Path to model adapter, if any.
            tokenizer_path (Optional[str]): Path to tokenizer, if different from model.
            model_type (str): Type of model ('causal' for generative models).
            max_length (int): Maximum input length for tokenization.
            device (str): Device to run model on ('auto', 'cpu', 'cuda').
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.logger = logger or setup_logger("src.generation.llm_generator")
        self.max_length = max_length
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))
        self.model_type = model_type

        try:
            self.model_loader = ModelLoader(
                model_name=model_path,
                model_type=model_type,
                adapter_path=adapter_path,
                tokenizer_path=tokenizer_path,
                device_map=self.device,
                max_length=self.max_length
            )
            self.model = self.model_loader.model
            tokenizer_source = tokenizer_path if tokenizer_path is not None else model_path
            self.tokenizer = create_and_configure_tokenizer(model=self.model, model_name=model_path, tokenizer_path=tokenizer_source)
            self.logger.info("Loaded model %s and tokenizer %s on %s", model_path, tokenizer_source, self.device)
        except Exception as e:
            self.logger.error("Failed to load model or tokenizer: %s", str(e))
            raise

    def format_prompt(self, query: str, contexts: str) -> str:
        """
        Format the input prompt with instructions for the LLM in Italian.

        Args:
            query (str): User query.
            contexts (str): Contexts from Augmenter (child and parent texts).

        Returns:
            str: Formatted prompt with instructions, query, and contexts.
        """
        instruction = (
            "Sei un assistente legale specializzato in diritto italiano. Rispondi alla query in italiano, "
            "utilizzando i contesti figlio e genitore forniti come riferimento. I contesti figlio forniscono "
            "dettagli specifici, mentre i contesti genitore offrono un contesto legale più ampio. Fornisci una "
            "risposta concisa e accurata senza ripetere verbatim la query o i contesti."
        )
        return f"{instruction}\n\n**Query**: {query}\n\n**Contesti**:\n{contexts}"

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate a response from a formatted prompt.

        Args:
            prompt (str): Input prompt containing query and contexts from Augmenter.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Generated response.
        """
        try:
            # Extract query from prompt (assuming format from Augmenter: "Query: ... \nContext: ...")
            query = prompt.split("\nContext:")[0].replace("Query: ", "").strip()
            contexts = "\nContext:".join(prompt.split("\nContext:")[1:]).strip()

            # Format prompt with instruction
            formatted_prompt = self.format_prompt(query, contexts)

            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.device)

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Decode response, skipping the input prompt
            print(outputs)
            print("_"*15)
            print(inputs["input_ids"].shape[1])
            print("_"*15)
            print(inputs["input_ids"].shape)
            print("_"*15)
            print(outputs[0].shape)
            print("_"*15)
            print(outputs[0][inputs["input_ids"].shape[1]:])
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response2 = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            print(f"Response: {response2}")
            self.logger.info("Generated response for query: %s...", query[:50])
            return response.strip()
        except Exception as e:
            self.logger.error("Generation failed for query '%s': %s", query[:50], str(e))
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    generator = LLMGenerator(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_type="causal",
        max_length=2048,
        device="auto"
    )
    prompt = """Query: Quali sono i requisiti per la residenza in Italia?
Context:
1. (Subject: courthouse, Chunk ID: 116876_chunk_0, Parent ID: 116876_parent_0, Score: 0.950)
Child Context: Per la residenza in Italia, è necessario un passaporto valido e un contratto di affitto o proprietà.
Parent Context: La residenza in Italia richiede la presentazione di documenti specifici al Comune di residenza...
2. (Subject: courthouse, Chunk ID: 116876_chunk_1, Parent ID: 116876_parent_0, Score: 0.920)
Child Context: I requisiti includono un'assicurazione sanitaria valida e un reddito sufficiente.
Parent Context: La residenza in Italia richiede la presentazione di documenti specifici al Comune di residenza..."""
    response = generator.generate(prompt, max_new_tokens=200)
    print(f"Response: {response}")