import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.utils.logging_utils import setup_logger

class Augmenter:
    """Handles augmentation of query and retrieved contexts for RAG generation."""

    def __init__(
        self,
        max_contexts: int = 5,
        max_context_length: int = 1000,
        max_parent_length: int = 2000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Augmenter with configuration parameters.

        Args:
            max_contexts (int): Maximum number of contexts to include in the prompt.
            max_context_length (int): Maximum character length for child chunk text.
            max_parent_length (int): Maximum character length for parent chunk text.
            logger (Optional[logging.Logger]): Logger instance, defaults to None.
        """
        self.max_contexts = max_contexts
        self.max_context_length = max_context_length
        self.max_parent_length = max_parent_length
        self.logger = logger or setup_logger("src.augmentation.augmenter")
        self.logger.info(
            "Initialized Augmenter with max_contexts=%d, max_context_length=%d, max_parent_length=%d",
            max_contexts, max_context_length, max_parent_length
        )

    def _load_parent_text(self, parent_file_path: str) -> str:
        """
        Load parent chunk text from file.

        Args:
            parent_file_path (str): Path to parent chunk file.

        Returns:
            str: Parent chunk text, or empty string if not found.
        """
        try:
            parent_file = Path(parent_file_path)
            if not parent_file.exists():
                self.logger.warning("Parent file not found: %s", parent_file_path)
                return ""
            with open(parent_file, "r", encoding="utf-8") as f:
                text = f.read()[:self.max_parent_length]
                if len(text) > self.max_parent_length:
                    text += "..."
                self.logger.debug("Loaded parent text from %s", parent_file_path)
                return text
        except Exception as e:
            self.logger.error("Failed to load parent text from %s: %s", parent_file_path, str(e))
            return ""

    def augment(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Augment the query with retrieved child and parent contexts for the language model.

        Args:
            query (str): User query (e.g., in Italian for legal documents).
            contexts (List[Dict[str, Any]]): Retrieved chunks with keys 'chunk_id', 'text', 'score',
                                            'parent_id', 'parent_file_path', and optional 'subject'.

        Returns:
            str: Formatted prompt combining query, child contexts, and parent contexts, or query alone if no valid contexts.
        """
        try:
            if not query or not isinstance(query, str):
                self.logger.error("Invalid query: %s", query)
                return f"Query: {query}\nContext: None"

            # Filter and select top contexts (unchanged)
            valid_contexts = [
                c for c in contexts
                if isinstance(c, dict) and "text" in c and c["text"].strip() and isinstance(c["text"], str)
                and "parent_file_path" in c and isinstance(c["parent_file_path"], str)
            ]
            if not valid_contexts:
                self.logger.warning("No valid contexts provided for query: %s", query[:50])
                return f"Query: {query}\nContext: None"

            sorted_contexts = sorted(valid_contexts, key=lambda x: x.get("score", 0.0), reverse=True)
            selected_contexts = sorted_contexts[:self.max_contexts]

            # New: Instructional preamble (customize as needed; Italian version below if queries are in IT)
            preamble = """Rispondi alla seguente query in italiano, basandoti principalmente sui contesti rilevanti forniti di seguito (estratti da documenti legali). 
    Usa la tua conoscenza generale per colmare lacune, ma cita sezioni specifiche dove possibile per garantire accuratezza. 
    Struttura la risposta in modo chiaro e conciso, indicando le fonti tra parentesi (es. [Contesto 1]).

    Query: {query}

    Contesti rilevanti:""".format(query=query)

            # Build context sections
            context_parts = []
            for i, context in enumerate(selected_contexts, 1):
                # Truncate child text (unchanged)
                child_text = context["text"][:self.max_context_length]
                if len(context["text"]) > self.max_context_length:
                    child_text += "..."

                # Load parent text (unchanged)
                parent_text = self._load_parent_text(context["parent_file_path"]) if context.get("parent_file_path") else ""

                # Metadata tag (compact)
                subject = context.get("subject", "tribunale")
                chunk_id = context.get("chunk_id", "sconosciuto")
                score = context.get("score", 0.0)
                metadata_tag = f"[Soggetto: {subject}, ID Chunk: {chunk_id}, Punteggio: {score:.3f}]"

                # Combined excerpt: Child as "focus," parent as "contesto ampliato" for natural flow
                excerpt = f"{i}. {metadata_tag}\nEstratto principale: {child_text}\nContesto ampliato: {parent_text[:self.max_parent_length] or 'Non disponibile'}..." if len(parent_text) > self.max_parent_length else parent_text or 'Non disponibile'

                context_parts.append(excerpt)

            # Assemble full prompt
            prompt = preamble + "\n\n".join(context_parts) + "\n\nRispondi ora alla query."

            self.logger.info("Augmented query with %d contexts for query: %s...", len(selected_contexts), query[:50])
            return prompt

        except Exception as e:
            self.logger.error("Augmentation failed for query '%s': %s", query[:50], str(e))
            return f"Query: {query}\nContext: None"

    def augment_old(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Augment the query with retrieved child and parent contexts for the language model.

        Args:
            query (str): User query (e.g., in Italian for legal documents).
            contexts (List[Dict[str, Any]]): Retrieved chunks with keys 'chunk_id', 'text', 'score',
                                            'parent_id', 'parent_file_path', and optional 'subject'.

        Returns:
            str: Formatted prompt combining query, child contexts, and parent contexts, or query alone if no valid contexts.
        """
        try:
            if not query or not isinstance(query, str):
                self.logger.error("Invalid query: %s", query)
                return f"Query: {query}\nContext: None"

            # Filter valid contexts
            valid_contexts = [
                c for c in contexts
                if isinstance(c, dict) and "text" in c and c["text"].strip() and isinstance(c["text"], str)
                and "parent_file_path" in c and isinstance(c["parent_file_path"], str)
            ]
            if not valid_contexts:
                self.logger.warning("No valid contexts provided for query: %s", query[:50])
                return f"Query: {query}\nContext: None"

            # Sort by score (descending, as per reranker) and select top-k
            sorted_contexts = sorted(valid_contexts, key=lambda x: x.get("score", 0.0), reverse=True)
            selected_contexts = sorted_contexts[:self.max_contexts]

            # Format prompt with child and parent contexts
            prompt_parts = [f"Query: {query}\nContext:"]
            for i, context in enumerate(selected_contexts, 1):
                # Truncate child context text
                child_text = context["text"][:self.max_context_length]
                if len(context["text"]) > self.max_context_length:
                    child_text += "..."

                # Load parent context
                parent_text = self._load_parent_text(context["parent_file_path"]) if context.get("parent_file_path") else ""

                # Include metadata
                subject = context.get("subject", "courthouse")
                chunk_id = context.get("chunk_id", "unknown")
                parent_id = context.get("parent_id", "unknown")
                score = context.get("score", 0.0)

                prompt_parts.append(
                    f"{i}. (Subject: {subject}, Chunk ID: {chunk_id}, Parent ID: {parent_id}, Score: {score:.3f})\n"
                    f"Child Context: {child_text}\n"
                    f"Parent Context: {parent_text or 'None'}"
                )

            prompt = "\n".join(prompt_parts)
            self.logger.info("Augmented query with %d contexts for query: %s...", len(selected_contexts), query[:50])
            return prompt
        except Exception as e:
            self.logger.error("Augmentation failed for query '%s': %s", query[:50], str(e))
            return f"Query: {query}\nContext: None"