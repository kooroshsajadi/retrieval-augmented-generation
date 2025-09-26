import hashlib
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import logging
from sentence_transformers import SentenceTransformer
from src.utils.logging_utils import setup_logger
from src.utils.ingestion.chunk_strategy import ChunkingStrategy
from src.utils.ingestion.sentence_based_chunking import create_sentence_based_chunks
from src.utils.ingestion.parent_chunking import ParentChildChunking
from src.utils.models.bi_encoders import EncoderModels
from src.utils.deduplication_utils import get_unique_text_files

class TextChunker:
    """Splits cleaned text into chunks for downstream processing using specified chunking strategy."""

    def __init__(
        self,
        input_dir: str = "data/cleaned_text",
        output_dir: str = "data/chunked_text",
        max_chunk_length: int = 2000,
        min_chunk_length: int = 10,
        max_tokens: int = 768,
        chunking_strategy: str = ChunkingStrategy.SENTENCE_BASED.value,
        embedder: Optional[SentenceTransformer] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TextChunker with configuration parameters.

        Args:
            input_dir (str): Directory containing cleaned text files.
            output_dir (str): Directory to save chunked text and metadata.
            max_chunk_length (int): Maximum characters per chunk (for parent) or words (for sentence-based).
            min_chunk_length (int): Minimum character count for valid chunks.
            max_tokens (int): Maximum tokens per chunk to respect embedding model limits.
            chunking_strategy (str): Chunking strategy ("sentence-based" or "parent").
            embedder (SentenceTransformer): Embedding model for token counting.
            logger (logging.Logger, optional): Logger instance.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self.max_tokens = max_tokens
        self.chunking_strategy = chunking_strategy
        self.embedder = embedder
        self.logger = logger or setup_logger("src.ingestion.text_chunker")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate chunking strategy
        if self.chunking_strategy not in [s.value for s in ChunkingStrategy]:
            raise ValueError(f"Unsupported chunking_strategy: {self.chunking_strategy}. Choose from {[s.value for s in ChunkingStrategy]}")

        # Validate embedder
        if self.embedder is None:
            raise ValueError("Embedder is required for token counting")

        # Sentence boundary regex for Italian
        self.sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙ])'

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex for Italian text.

        Args:
            text (str): Input text to split.

        Returns:
            List[str]: List of sentences.
        """
        if not text:
            return []
        try:
            sentences = re.split(self.sentence_pattern, text.strip())
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.error("Sentence splitting failed: %s", str(e))
            return []

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the embedder's tokenizer.

        Args:
            text (str): Input text.

        Returns:
            int: Number of tokens.
        """
        try:
            tokens = self.embedder.tokenizer(text, return_tensors="pt", truncation=False).input_ids
            return tokens.shape[1]
        except Exception as e:
            self.logger.error(f"Token counting failed: %s", str(e))
            return 0

    def is_valid_chunk(self, text: str) -> bool:
        """
        Validate if a chunk is meaningful.

        Args:
            text (str): Chunk text to validate.

        Returns:
            bool: True if chunk is valid (sufficient length and content).
        """
        if not text or len(text.strip()) < self.min_chunk_length:
            return False
        # Check for Italian diacritics or legal terms
        if re.search(r'[àèìòù]', text, re.IGNORECASE) or re.search(r'\b(legge|decreto|articolo)\b', text, re.IGNORECASE):
            return True
        # Fallback: At least 3 words
        words = text.split()
        return len(words) >= 3

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single text file, chunk it, and save results.

        Args:
            file_path (Path): Path to the input text file.

        Returns:
            Dict[str, Any]: Chunking result with chunks and metad
        """
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "chunks": [],
            "is_valid": False,
            "error": None,
            "original_length": 0,
            "chunk_count": 0
        }
        self.logger.info("Chunking text file: %s with strategy: %s", file_path, self.chunking_strategy)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            result["original_length"] = len(text)

            if self.chunking_strategy == ChunkingStrategy.SENTENCE_BASED.value:
                sentences = self.split_into_sentences(text)
                chunks = create_sentence_based_chunks(
                    sentences=sentences,
                    max_chunk_words=self.max_chunk_length,
                    max_tokens=self.max_tokens,
                    count_tokens=self.count_tokens,
                    is_valid_chunk=self.is_valid_chunk
                )
            else:  # parent
                parent_chunker = ParentChildChunking(
                    text_dir=str(self.input_dir),
                    max_tokens=self.max_tokens,
                    max_chunk_length=self.max_chunk_length
                )
                file_stem = file_path.name.rsplit(".", 1)[0]
                chunks = parent_chunker.chunk(
                    file_path=str(file_path),
                    count_tokens=self.count_tokens,
                    is_valid_chunk=self.is_valid_chunk,
                    file_stem=file_stem
                )

            result["chunks"] = chunks
            result["chunk_count"] = len(chunks)
            result["is_valid"] = any(chunk["is_valid"] for chunk in chunks)
            if not result["is_valid"]:
                result["error"] = "No valid chunks created (too short or lacks meaningful content)"
                self.logger.warning(result["error"])
            self.save_chunks(file_path.name, chunks)
            return result
        except Exception as e:
            self.logger.error("Failed to process %s: %s", file_path, str(e))
            result["error"] = str(e)
            return result

    def save_chunks(self, file_name: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Save each chunk's text to output directory with metadata.

        Args:
            file_name (str): Name of original file.
            chunks (List[Dict[str, Any]]): Chunk dictionary list.
        """
        # file_stem = file_name.rsplit(".", 1)[0]
        for i, chunk in enumerate(chunks, 1):
            # chunk_type = chunk.get("chunk_type")
            # file_name = f"{file_stem}_chunk_{i:03d}" if chunk_type == "child" else f"{file_stem}_parent_{i:03d}"
            chunk_file_name = chunk["file_name"]
            chunk_file_path = self.output_dir / f"{chunk_file_name}.txt"
            try:
                with open(chunk_file_path, "w", encoding="utf-8") as f:
                    f.write(chunk["text"])
                self.logger.info("Saved chunk to %s", chunk_file_path)
            except Exception as e:
                self.logger.error("Failed to save chunk to %s: %s", chunk_file_path, str(e))

    def process_directory(self) -> None:
        """
        Process all text files in the input directory and save accumulated metadata once.
        """
        # text_files = list(self.input_dir.glob("*.txt"))
        text_files = get_unique_text_files(self.input_dir)
        if not text_files:
            self.logger.warning("No text files found in %s", self.input_dir)
            return
        self.logger.info("Processing %d text files in %s with strategy %s", len(text_files), self.input_dir, self.chunking_strategy)
        processed_files = 0
        metadata_collection = []
        for file_path in text_files:
            result = self.process_file(file_path)
            file_stem = file_path.name.rsplit(".", 1)[0]
            chunks_metadata = []
            parent_chunk_ids = {}  # Map hashed parent_id to parent chunk_id
            chunk_counter = 1

            for chunk in result["chunks"]:
                chunk_type = chunk.get("chunk_type")
                # chunk_id = f"{file_stem}_chunk_{chunk_counter:03d}" if chunk_type == "child" else f"{file_stem}_parent_{chunk_counter:03d}"
                chunk_id = chunk["parent_id"]
                if chunk_type == "parent":
                    parent_chunk_ids[chunk["parent_id"]] = chunk_id
                chunk_path = Path(self.output_dir) / f"{chunk['file_name']}.txt"
                chunks_metadata.append({
                    "file_name": chunk["file_name"],
                    "file_path": str(chunk_path),
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk.get("chunk_type"),
                    "word_count": chunk["word_count"],
                    "char_length": chunk["char_length"],
                    "token_count": chunk["token_count"],
                    "is_valid": chunk["is_valid"],
                    "parent_id": chunk["parent_id"],
                    "parent_file_name": chunk.get("parent_file_name"),
                    "parent_file_path": str(file_path) if chunk_type == "parent" else str(self.output_dir / f"{chunk['parent_file_name']}.txt")
                })
                chunk_counter += 1

            # Update parent_id for child chunks to match parent chunk_id
            # for meta in chunks_metadata:
            #     if meta["chunk_type"] == "child" and meta["parent_id"] in parent_chunk_ids:
            #         meta["parent_id"] = parent_chunk_ids[meta["parent_id"]]

            file_metadata = {
                "file_name": result["file_name"],
                "file_path": result["file_path"],
                "file_id": hashlib.md5(file_stem.encode()).hexdigest(),
                "original_length": result["original_length"],
                "is_valid": result["is_valid"],
                "error": result["error"],
                "chunk_count": result["chunk_count"],
                "chunking_strategy": self.chunking_strategy,
                "chunks_metadata": chunks_metadata
            }
            metadata_collection.append(file_metadata)
            processed_files += 1

        self.logger.info("Processed %d/%d text files", processed_files, len(text_files))
        summary_file = f"data/metadata/leggi_area_3_chunks_{self.chunking_strategy}.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(metadata_collection, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved chunking summary to %s", summary_file)
        except Exception as e:
            self.logger.error("Failed to save chunking summary: %s", str(e))

if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        embedder = SentenceTransformer(EncoderModels.ITALIAN_LEGAL_BERT_SC.value)
        chunker = TextChunker(
            input_dir=config['cleaned_texts'].get('leggi_area_3', 'data/leggi_area_3_cleaned_texts'),
            output_dir=config['chunks'].get('leggi_area_3', 'data/chunks/leggi_area_3_chunks'),
            max_chunk_length=2000,
            min_chunk_length=10,
            max_tokens=768,
            chunking_strategy=config.get('chunking_strategy', ChunkingStrategy.PARENT.value),
            embedder=embedder
        )
        chunker.process_directory()
        print("Text chunking completed.")
    except Exception as e:
        print(f"Error during text chunking: {e}")