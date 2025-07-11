from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class Generator(ABC):
    """Abstract base class for generation models, defining the interface for text generation."""
    
    @abstractmethod
    def generate(self, prompt: str, contexts: List[str]) -> str:
        """Generate response given a prompt and retrieved contexts."""
        pass

class HuggingFaceGenerator(Generator):
    """Implements text generation using a Hugging Face pre-trained language model."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_length = 512

    def generate(self, prompt: str, contexts: List[str]) -> str:
        # Combine contexts and prompt
        context_str = "\n".join(contexts)
        full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}\nAnswer:"
        
        # Tokenize and generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        
        # Decode response
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)