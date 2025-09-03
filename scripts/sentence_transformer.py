import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion.text_chunker import TextChunker
from src.utils.logging_utils import setup_logger

class SentenceTransformerEmbedder:
    """Generates embeddings for user queries or file-extracted text in RAG pipeline."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        output_dir: str = "data/embeddings",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize SentenceTransformerEmbedder.

        Args:
            model_name (str): SentenceTransformer model name.
            output_dir (str): Directory to save embeddings and metadata.
            chunk_size (int): Maximum size of text chunks for file text.
            chunk_overlap (int): Overlap between text chunks.
            logger (logging.Logger, optional): Logger instance.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.logger = setup_logger("sentence_transformer")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SentenceTransformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info("Loaded SentenceTransformer model: %s", model_name)
        except Exception as e:
            self.logger.error("Failed to load SentenceTransformer model: %s", str(e))
            raise

        # Initialize TextChunker for file text
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            logger=self.logger
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text (str): Text to embed (query or chunk).

        Returns:
            np.ndarray: Embedding vector, or empty array on failure.
        """
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            self.logger.debug("Generated embedding for text (length: %d)", len(text))
            return embedding
        except Exception as e:
            self.logger.error("Embedding generation failed: %s", str(e))
            return np.array([])

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Generate embedding for a user query or command.

        Args:
            query (str): User query or command text.

        Returns:
            Dict[str, Any]: Result with query, embedding, and status.
        """
        result = {
            "query": query,
            "embedding": None,
            "is_valid": False,
            "error": None
        }

        self.logger.info("Processing query: %s", query[:50])
        try:
            embedding = self.generate_embedding(query)
            if embedding.size == 0:
                result["error"] = "Failed to generate embedding"
                self.logger.error(result["error"])
                return result

            result["embedding"] = embedding
            result["is_valid"] = True
            self.logger.info("Successfully generated embedding for query")
            return result
        except Exception as e:
            result["error"] = str(e)
            self.logger.error("Query embedding failed: %s", str(e))
            return result

    def process_file(self, file_path: str, extracted_text: str = None) -> Dict[str, Any]:
        """
        Generate embeddings for text extracted from a file.

        Args:
            file_path (str): Path to the original file (for metadata).
            extracted_text (str, optional): Pre-extracted text; if None, read from data/texts/.

        Returns:
            Dict[str, Any]: Result with chunks, embeddings, and metadata.
        """
        file_path = Path(file_path)
        result = {
            "file_path": file_path.as_posix(),
            "file_name": file_path.name,
            "is_valid": False,
            "error": None,
            "chunk_embeddings": []
        }

        self.logger.info("Processing file: %s", file_path)
        try:
            # Read extracted text if not provided
            if extracted_text is None:
                text_file = Path("data/texts") / f"{file_path.stem}.txt"
                if not text_file.exists():
                    result["error"] = f"Extracted text file not found: {text_file}"
                    self.logger.error(result["error"])
                    return result
                with open(text_file, "r", encoding="utf-8") as f:
                    extracted_text = f.read()

            # Chunk the text
            chunks = self.chunker.chunk_text(extracted_text)
            if not chunks:
                result["error"] = "No valid chunks generated"
                self.logger.error(result["error"])
                return result

            # Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_chunk_{i}"
                embedding = self.generate_embedding(chunk)
                if embedding.size == 0:
                    self.logger.warning("Empty embedding for chunk %s", chunk_id)
                    continue

                # Save embedding
                embedding_file = self.output_dir / f"{chunk_id}.npy"
                try:
                    np.save(embedding_file, embedding)
                    self.logger.info("Saved embedding to %s", embedding_file)
                except Exception as e:
                    self.logger.error("Failed to save embedding to %s: %s", embedding_file, str(e))
                    continue

                # Store metadata
                result["chunk_embeddings"].append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "embedding_file": f"{chunk_id}.npy",
                    "is_valid": True
                })

            result["is_valid"] = len(result["chunk_embeddings"]) > 0
            if not result["is_valid"]:
                result["error"] = "No valid embeddings generated"
                self.logger.warning(result["error"])

            # Save metadata
            summary_file = self.output_dir / "embeddings_summary.json"
            existing_results = []
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            existing_results.append(result)
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            self.logger.info("Updated embeddings summary: %s", summary_file)

            return result
        except Exception as e:
            result["error"] = str(e)
            self.logger.error("File embedding failed: %s", str(e))
            return result

if __name__ == "__main__":
    # Example usage
    embedder = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-large",
        output_dir="data/embeddings",
        chunk_size=512,
        chunk_overlap=50
    )

    # Test with a query
    query = "Quali sono i requisiti per la residenza in Italia?"
    query_result = embedder.process_query(query)
    print(f"Query result: {query_result}")

    # Test with a file
    file_path = "data/source/sample.pdf"
    file_result = embedder.process_file(file_path)
    print(f"File result: {file_result}")