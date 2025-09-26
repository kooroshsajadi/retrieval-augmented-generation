import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion.text_chunker import TextChunker
from src.utils.models.bi_encoders import EncoderModels
import yaml
from src.utils.logging_utils import setup_logger
import torch
from src.utils.ingestion.chunk_strategy import ChunkingStrategy

class EmbeddingGenerator:
    """Generates vector embeddings for chunked text using SentenceTransformer."""

    def __init__(
        self,
        input_dir: str = "data/chunked_text",
        output_dir: str = "data/embeddings",
        max_chunk_length: int = 2000,
        min_chunk_length: int = 10,
        chunking_info_path: Optional[str] = None,
        model_name: str = "dlicari/Italian-Legal-BERT-SC",
        logger: Optional[logging.Logger] = None,
        chunking_strategy=ChunkingStrategy.PARENT.value,
        metadata_path: Optional[str] = "data/metadata/embeddings_leggi_area_3.json",
    ):
        """
        Initialize EmbeddingGenerator with configuration parameters.

        Args:
            input_dir (str): Directory containing chunked text files.
            output_dir (str): Directory to save embeddings and metadata.
            chunking_info_path (str): Path to chunking metadata file.
            model_name (str): SentenceTransformer model name.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if chunking_info_path is not None:
            self.chunking_info_path = Path(chunking_info_path)
        self.model_name = model_name
        self.logger = logger or setup_logger("src.embeddings.sentence_transformer")
        self.chunking_strategy = chunking_strategy
        self.metadata_path = Path(metadata_path) if metadata_path else self.output_dir / "embeddings_summary.json"

        # Initialize model
        try:
            device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
            self.logger.info("Loaded SentenceTransformer model: %s", model_name)
            self.logger.info("Using device: %s", device)
        except Exception as e:
            self.logger.error("Failed to initialize SentenceTransformer model: %s", str(e))
            raise

        # Initialize TextChunker for file text
        self.chunker = TextChunker(
            max_chunk_length=max_chunk_length,
            min_chunk_length=min_chunk_length,
            embedder=self.model,
            logger=self.logger
        )
        
        # Ensure output directory exists.
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
    
    def process_file(self, file_path: str, extracted_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate embeddings for text extracted from a file.

        Args:
            file_path (str): Path to the original file (for metadata).
            extracted_text (str, optional): Pre-extracted text; if None, read from data/texts/.

        Returns:
            Dict[str, Any]: Result with chunks, embeddings, and metadata.
        """
        sample_path = Path(file_path)
        result = {
            "file_path": sample_path.as_posix(),
            "file_name": sample_path.name,
            "is_valid": False,
            "error": None,
            "chunk_embeddings": []
        }

        self.logger.info("Processing file: %s", file_path)
        try:
            # Read extracted text if not provided
            if extracted_text is None:
                text_file = Path("data/texts") / f"{sample_path.stem}.txt"
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
                chunk_id = f"{sample_path.stem}_chunk_{i}"
                embedding = self.generate_embedding(chunk["text"])
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
                    "text": chunk["text"],
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

    def process_internal_file(self, file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process child chunks for a single file and generate embeddings.

        Args:
            file_metadata (Dict[str, Any]): Metadata from chunking_summary.json.

        Returns:
            Dict[str, Any]: Embedding result with metadata.
        """
        result = {
            "file_path": file_metadata["file_path"],
            "file_name": file_metadata["file_name"],
            "file_id": file_metadata["file_id"],
            "is_valid": False,
            "error": None,
            "chunk_embeddings": []
        }

        self.logger.info("Processing file: %s", file_metadata["file_path"])
        try:
            for chunk_meta in file_metadata["chunks_metadata"]:
                if chunk_meta["chunk_type"] != "child":
                    self.logger.debug("Skipping non-child chunk: %s", chunk_meta["chunk_id"])
                    continue
                if not chunk_meta["is_valid"]:
                    self.logger.warning("Skipping invalid chunk: %s", chunk_meta["chunk_id"])
                    continue

                chunk_file = Path(chunk_meta["file_path"])
                if not chunk_file.exists():
                    self.logger.warning("Chunk file not found: %s", chunk_file)
                    continue

                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read()

                embedding = self.generate_embedding(chunk_text)
                if embedding.size == 0:
                    self.logger.warning("Empty embedding for chunk: %s", chunk_meta["chunk_id"])
                    continue

                # Save embedding file using chunk_id
                embedding_file = self.output_dir / f"{chunk_meta['chunk_id']}.npy"
                try:
                    np.save(embedding_file, embedding)
                    self.logger.info("Saved embedding to %s", embedding_file)
                except Exception as e:
                    self.logger.error("Failed to save embedding to %s: %s", embedding_file, str(e))
                    result["error"] = str(e)

                # Accumulate metadata with parent information
                result["chunk_embeddings"].append({
                    "file_name": chunk_meta["file_name"],
                    "file_path": chunk_meta["file_path"],
                    "chunk_id": chunk_meta["chunk_id"],
                    "word_count": chunk_meta["word_count"],
                    "char_length": chunk_meta["char_length"],
                    "token_count": chunk_meta["token_count"],
                    "embedding_file": f"{chunk_meta['chunk_id']}.npy",
                    "is_valid": True,
                    "parent_id": chunk_meta["parent_id"],
                    "parent_file_name": chunk_meta["parent_file_name"],
                    "parent_file_path": chunk_meta["parent_file_path"]
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

        self.logger.info("Processing files in %s based on metadata.", self.input_dir)
        processed_files = 0
        results = []

        if self.chunking_strategy == ChunkingStrategy.PARENT.value:
            for file_metadata in metadata:
                if not file_metadata["is_valid"]:
                    self.logger.warning("Skipping invalid file: %s", file_metadata["file_path"])
                    continue
                result = self.process_internal_file(file_metadata)
                results.append(result)
                processed_files += 1
        else:
            for file_metadata in metadata:
                if not file_metadata["is_valid"]:
                    self.logger.warning("Skipping invalid file: %s", file_metadata["file_path"])
                    continue
                result = self.process_file(file_metadata["file_path"])
                results.append(result)
                processed_files += 1

        self.logger.info("Processed %d/%d files", processed_files, len(metadata))

        # Save all metadata in a single summary file
        summary_file = str(self.metadata_path)
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
        summary_file = self.metadata_path
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
            input_dir=config['chunks'].get('leggi_area_3', 'data/chunks/leggi_area_3_chunks'),
            output_dir=config['embeddings'].get('leggi_area_3', 'data/embeddings/leggi_area_3_embeddings'),
            chunking_info_path=config['metadata'].get('leggi_area_3', 'data/metadata/leggi_area_3_chunks_parent.json'),
            metadata_path="data/metadata/embeddings_leggi_area_3.json",
            model_name=EncoderModels.ITALIAN_LEGAL_BERT_SC.value,
            chunking_strategy=ChunkingStrategy.PARENT.value
        )
        generator.process_directory()
        print("Embedding generation completed.")
    except Exception as e:
        print(f"Error during embedding generation: {e}")