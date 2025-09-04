import logging
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import setup_logger

class Augmenter:
    """Handles augmentation of query and retrieved contexts for RAG generation."""

    def __init__(
        self,
        max_contexts: int = 5,
        max_context_length: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Augmenter with configuration parameters.

        Args:
            max_contexts (int): Maximum number of contexts to include in the prompt.
            max_context_length (int): Maximum character length per context to prevent overly long prompts.
            logger (Optional[logging.Logger]): Logger instance, defaults to None.
        """
        self.max_contexts = max_contexts
        self.max_context_length = max_context_length
        self.logger = logger or setup_logger("augmenter")
        self.logger.info("Initialized Augmenter with max_contexts=%d, max_context_length=%d", max_contexts, max_context_length)

    def augment(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Augment the query with retrieved contexts to create a prompt for the language model.

        Args:
            query (str): User query (e.g., in Italian for legal documents).
            contexts (List[Dict[str, Any]]): Retrieved chunks with keys 'chunk_id', 'text', 'distance', and optional 'subject'.

        Returns:
            str: Formatted prompt combining query and contexts, or query alone if no valid contexts.
        """
        try:
            if not query or not isinstance(query, str):
                self.logger.error("Invalid query: %s", query)
                return f"Query: {query}\nContext: None"

            # Filter valid contexts and sort by distance (ascending)
            valid_contexts = [
                c for c in contexts
                if isinstance(c, dict) and "text" in c and c["text"].strip() and isinstance(c["text"], str)
            ]
            if not valid_contexts:
                self.logger.warning("No valid contexts provided for query: %s", query[:50])
                return f"Query: {query}\nContext: None"

            sorted_contexts = sorted(valid_contexts, key=lambda x: x.get("distance", float("inf")))
            selected_contexts = sorted_contexts[:self.max_contexts]

            # Format prompt
            prompt_parts = [f"Query: {query}\nContext:"]
            for i, context in enumerate(selected_contexts, 1):
                # Truncate context text if too long
                context_text = context["text"][:self.max_context_length]
                if len(context["text"]) > self.max_context_length:
                    context_text += "..."
                # Include metadata if available (e.g., subject)
                subject = context.get("subject", "unknown")
                prompt_parts.append(f"{i}. ({subject}) {context_text}")
            prompt = "\n".join(prompt_parts)

            self.logger.info("Augmented query with %d contexts for query: %s...", len(selected_contexts), query[:50])
            return prompt
        except Exception as e:
            self.logger.error("Augmentation failed for query '%s': %s", query[:50], str(e))
            return f"Query: {query}\nContext: None"

if __name__ == "__main__":
    # Example usage
    augmenter = Augmenter(max_contexts=2, max_context_length=500)
    query = "Quali sono i requisiti per la residenza in Italia?"
    contexts = [
        {"chunk_id": "116876_chunk_0", "text": "Per la residenza in Italia, è necessario un passaporto valido e un contratto di affitto o proprietà.", "distance": 0.12, "subject": "courthouse"},
        {"chunk_id": "116876_chunk_1", "text": "I requisiti includono un'assicurazione sanitaria valida e un reddito sufficiente.", "distance": 0.15, "subject": "courthouse"}
    ]
    prompt = augmenter.augment(query, contexts)
    print(f"Prompt:\n{prompt}")