import json
import os
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.models.bi_encoders import EncoderModels
import yaml
from src.utils.logging_utils import setup_logger
    
class EmbeddingGenerator:
    """Generates vector embeddings for chunked text using SentenceTransformer."""

    def __init__(
        self,
        input_dir: str = "data/chunked_text",
        output_dir: str = "data/embeddings",
        chunking_info_path: str = "data/metadata/chunking_prefettura_v1.2.json",
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
        self.chunking_info_path = Path(chunking_info_path)
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

        if not self.chunking_info_path.exists():
            self.logger.error("Chunking metadata file not found: %s", self.chunking_info_path)
            raise FileNotFoundError(f"Chunking metadata file not found: {self.chunking_info_path}")

        try:
            with open(self.chunking_info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load chunking metadata: %s", str(e))
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

                # Save embedding file
                embedding_file = self.output_dir / f"{chunk_meta['chunk_id']}.npy"
                try:
                    np.save(embedding_file, embedding)
                    self.logger.info("Saved embedding to %s", embedding_file)
                except Exception as e:
                    self.logger.error("Failed to save embedding to %s: %s", embedding_file, str(e))
                    result["error"] = str(e)

                # Accumulate minimal metadata (no embedding itself)
                result["chunk_embeddings"].append({
                    "chunk_id": chunk_meta["chunk_id"],
                    "word_count": chunk_meta["word_count"],
                    "char_length": chunk_meta["char_length"],
                    "embedding_file": f"{chunk_meta['chunk_id']}.npy",
                    "is_valid": True
                })

            result["is_valid"] = len(result["chunk_embeddings"]) > 0
            if not result["is_valid"]:
                result["error"] = "No valid embeddings generated"
                self.logger.warning(result["error"])

            return result

        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_metadata["file_path"], str(e))
            result["error"] = str(e)
            return result

    def process_directory(self) -> None:
        """
        Process all chunked files in the input directory.
        Save all metadata in a single summary file.
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

        # Save all metadata in a single summary file
        summary_file = "data/metadata/embeddings_prefettura_v1.3.json"
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
        summary_file = "data/metadata/embeddings_prefettura_v1.3.json"
        if not os.path.exists(summary_file):
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
            input_dir=config['chunks'].get('prefettura_v1.3', 'data/chunks/prefettura_v1.3_chunks'),
            output_dir=config['embeddings'].get('prefettura_v1.3', 'data/embeddings/prefettura_v1.3_embeddings'),
            chunking_info_path=config['metadata'].get('chunking_prefettura_v1.3', 'data/metadata/chunking_prefettura_v1.3_chunks.json'),
            model_name=EncoderModels.ITALIAN_LEGAL_BERT_SC.value
        )
        generator.process_directory()
        print("Embedding generation completed.")
    except Exception as e:
        print(f"Error during embedding generation: {e}")
