import json
from typing import Dict, List, Any
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from src.utils.logging_utils import setup_logger

class EmbeddingGenerator:
    """Generates vector embeddings for chunked text using SentenceTransformer."""

    def __init__(
        self,
        input_dir: str = "data/chunked_text",
        output_dir: str = "data/embeddings",
        model_name: str = "intfloat/multilingual-e5-large",
    ):
        """
        Initialize EmbeddingGenerator with configuration parameters.

        Args:
            input_dir (str): Directory containing chunked text files.
            output_dir (str): Directory to save embeddings and metadata.
            model_name (str): SentenceTransformer model name.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.logger = setup_logger("sentence_transformer")

        # Initialize model
        self.model = SentenceTransformer(model_name)
        self.logger.info("Loaded SentenceTransformer model: %s", model_name)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_chunk_metadata(self) -> List[Dict[str, Any]]:
        """
        Load chunking metadata from summary file.

        Returns:
            List[Dict[str, Any]]: List of chunking result dictionaries.
        """
        summary_file = self.input_dir / "chunking_summary.json"
        if not summary_file.exists():
            self.logger.error("Chunking summary file not found: %s", summary_file)
            raise FileNotFoundError(f"Chunking summary file not found: {summary_file}")

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load chunking summary: %s", str(e))
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text chunk.

        Args:
            text (str): Text chunk to embed.

        Returns:
            np.ndarray: Embedding vector.
        """
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            self.logger.debug("Generated embedding for text (length: %d)", len(text))
            return embedding
        except Exception as e:
            self.logger.error("Embedding generation failed: %s", str(e))
            return np.array([])

    def process_file(self, file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process chunks for a single file and generate embeddings.

        Args:
            file_metadata (Dict[str, Any]): Metadata from chunking_summary.json.

        Returns:
            Dict[str, Any]: Embedding result with metadata.
        """
        result = {
            "file_path": file_metadata["file_path"],
            "file_name": file_metadata["file_name"],
            "is_valid": False,
            "error": None,
            "chunk_embeddings": []
        }

        self.logger.info("Processing file: %s", file_metadata["file_path"])
        try:
            for chunk_meta in file_metadata["chunks_metadata"]:
                if not chunk_meta["is_valid"]:
                    self.logger.warning("Skipping invalid chunk: %s", chunk_meta["chunk_id"])
                    continue

                chunk_file = self.input_dir / f"{chunk_meta['chunk_id']}.txt"
                if not chunk_file.exists():
                    self.logger.warning("Chunk file not found: %s", chunk_file)
                    continue

                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read()

                embedding = self.generate_embedding(chunk_text)
                if embedding.size == 0:
                    self.logger.warning("Empty embedding for chunk: %s", chunk_meta["chunk_id"])
                    continue

                result["chunk_embeddings"].append({
                    "chunk_id": chunk_meta["chunk_id"],
                    "word_count": chunk_meta["word_count"],
                    "char_length": chunk_meta["char_length"],
                    "embedding_file": f"{chunk_meta['chunk_id']}.npy",
                    "is_valid": True
                })

                # Save embedding
                embedding_file = self.output_dir / f"{chunk_meta['chunk_id']}.npy"
                try:
                    np.save(embedding_file, embedding)
                    self.logger.info("Saved embedding to %s", embedding_file)
                except Exception as e:
                    self.logger.error("Failed to save embedding to %s: %s", embedding_file, str(e))
                    result["error"] = str(e)

            result["is_valid"] = len(result["chunk_embeddings"]) > 0
            if not result["is_valid"]:
                result["error"] = "No valid embeddings generated"
                self.logger.warning(result["error"])

            # Save metadata
            self.save_metadata(result)
            return result

        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_metadata["file_path"], str(e))
            result["error"] = str(e)
            return result

    def save_metadata(self, result: Dict[str, Any]) -> None:
        """
        Save embedding metadata to output directory.

        Args:
            result (Dict[str, Any]): Embedding result with metadata.
        """
        file_name = result["file_name"].rsplit(".", 1)[0]
        metadata_file = self.output_dir / f"{file_name}_metadata.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved embedding metadata to %s", metadata_file)
        except Exception as e:
            self.logger.error("Failed to save metadata to %s: %s", metadata_file, str(e))
            result["error"] = str(e)

    def process_directory(self) -> None:
        """
        Process all chunked files in the input directory.
        """
        metadata = self.load_chunk_metadata()
        if not metadata:
            self.logger.warning("No chunking metadata found. Skipping processing.")
            return

        self.logger.info("Processing %d files in %s", len(metadata), self.input_dir)
        processed_files = 0
        results = []

        for file_metadata in metadata:
            if not file_metadata["is_valid"]:
                self.logger.warning("Skipping invalid file: %s", file_metadata["file_path"])
                continue
            result = self.process_file(file_metadata)
            results.append(result)
            processed_files += 1

        self.logger.info("Processed %d/%d files", processed_files, len(metadata))

        # Save summary metadata
        summary_file = self.output_dir / "embeddings_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved embeddings summary to %s", summary_file)
        except Exception as e:
            self.logger.error("Failed to save embeddings summary: %s", str(e))

    def get_embedding_results(self) -> List[Dict[str, Any]]:
        """
        Load embedding results from summary file.

        Returns:
            List[Dict[str, Any]]: List of embedding result dictionaries.
        """
        summary_file = self.output_dir / "embeddings_summary.json"
        if not summary_file.exists():
            self.logger.warning("Embeddings summary file not found: %s", summary_file)
            return []

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load embeddings summary: %s", str(e))
            return []

if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        generator = EmbeddingGenerator(
            input_dir=config['chunks']['prefettura_v1'],
            output_dir=config['embeddings']['prefettura_v1'],
            model_name=config.get('embedding_model', 'intfloat/multilingual-e5-large')
        )
        generator.process_directory()
        print("Embedding generation completed.")
    except Exception as e:
        print(f"Error during embedding generation: {e}")