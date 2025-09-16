import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import logging
from transformers import AutoTokenizer
from src.utils.logging_utils import setup_logger
from src.utils.ingestion.chunk_strategy import ChunkingStrategy
from src.utils.ingestion.sentence_based_chunking import create_sentence_based_chunks
from src.utils.ingestion.parent_chunking import create_parent_chunks
from src.utils.models.bi_encoders import EncoderModels

class TextChunker:
    """Splits cleaned text into chunks for downstream processing using specified chunking strategy."""

    def __init__(
        self,
        input_dir: str = "data/cleaned_text",
        output_dir: str = "data/chunked_text",
        max_chunk_words: int = 500,
        min_chunk_length: int = 10,
        max_tokens: int = 512,
        chunking_strategy: str = ChunkingStrategy.SENTENCE_BASED.value,
        tokenizer_name: str = "intfloat/multilingual-e5-large-instruct",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TextChunker with configuration parameters.

        Args:
            input_dir (str): Directory containing cleaned text files.
            output_dir (str): Directory to save chunked text and metadata.
            max_chunk_words (int): Maximum words per chunk (for sentence-based) or child chunk (for parent).
            min_chunk_length (int): Minimum character count for valid chunks.
            max_tokens (int): Maximum tokens per chunk to respect embedding model limits.
            chunking_strategy (str): Chunking strategy ("sentence-based" or "parent").
            tokenizer_name (str): Name of tokenizer for token counting.
            logger (logging.Logger, optional): Logger instance.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_chunk_words = max_chunk_words
        self.min_chunk_length = min_chunk_length
        self.max_tokens = max_tokens
        self.chunking_strategy = chunking_strategy
        self.logger = logger or setup_logger("src.ingestion.text_chunker")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate chunking strategy
        if self.chunking_strategy not in [s.value for s in ChunkingStrategy]:
            raise ValueError(f"Unsupported chunking_strategy: {self.chunking_strategy}. Choose from {[s.value for s in ChunkingStrategy]}")

        # Load tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
            self.logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer {tokenizer_name}: {str(e)}")
            raise

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
        Count tokens in text using the tokenizer.

        Args:
            text (str): Input text.

        Returns:
            int: Number of tokens.
        """
        try:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=False).input_ids
            return tokens.shape[1]
        except Exception as e:
            self.logger.error(f"Token counting failed: {str(e)}")
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
            Dict[str, Any]: Chunking result with chunks and metadata.
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
            sentences = self.split_into_sentences(text)

            # Choose chunking strategy
            if self.chunking_strategy == ChunkingStrategy.SENTENCE_BASED.value:
                chunks = create_sentence_based_chunks(
                    sentences=sentences,
                    max_chunk_words=self.max_chunk_words,
                    max_tokens=self.max_tokens,
                    count_tokens=self.count_tokens,
                    is_valid_chunk=self.is_valid_chunk
                )
            else:  # parent
                chunks = create_parent_chunks(
                    text=text,
                    sentences=sentences,
                    max_tokens=self.max_tokens,
                    count_tokens=self.count_tokens,
                    is_valid_chunk=self.is_valid_chunk
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
        file_stem = file_name.rsplit(".", 1)[0]
        for i, chunk in enumerate(chunks, 1):
            chunk_type = chunk.get("chunk_type", "child")
            chunk_id = f"{file_stem}_chunk_{i:03d}" if chunk_type == "child" else f"{file_stem}_parent_{i:03d}"
            chunk_file = self.output_dir / f"{chunk_id}.txt"
            try:
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunk["text"])
                self.logger.info("Saved chunk to %s", chunk_file)
            except Exception as e:
                self.logger.error("Failed to save chunk to %s: %s", chunk_file, str(e))

    def process_directory(self) -> None:
        """
        Process all text files in the input directory and save accumulated metadata once.
        """
        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning("No text files found in %s", self.input_dir)
            return
        self.logger.info("Processing %d text files in %s with strategy %s", len(text_files), self.input_dir, self.chunking_strategy)
        processed_files = 0
        metadata_collection = []
        for file_path in text_files:
            result = self.process_file(file_path)
            file_metadata = {
                "file_path": result["file_path"],
                "file_name": result["file_name"],
                "is_valid": result["is_valid"],
                "error": result["error"],
                "original_length": result["original_length"],
                "chunk_count": result["chunk_count"],
                "chunking_strategy": self.chunking_strategy,
                "chunks_metadata": [
                    {
                        "chunk_id": f"{result['file_name'].rsplit('.', 1)[0]}_chunk_{i:03d}" if chunk.get("chunk_type", "child") == "child" else f"{result['file_name'].rsplit('.', 1)[0]}_parent_{i:03d}",
                        "word_count": chunk["word_count"],
                        "char_length": chunk["char_length"],
                        "token_count": chunk["token_count"],
                        "is_valid": chunk["is_valid"],
                        "parent_id": chunk["parent_id"],
                        "chunk_type": chunk.get("chunk_type", "child")
                    }
                    for i, chunk in enumerate(result["chunks"], 1)
                ]
            }
            metadata_collection.append(file_metadata)
            processed_files += 1
        self.logger.info("Processed %d/%d text files", processed_files, len(text_files))
        summary_file = f"data/metadata/chunking_prefettura_v1.3_chunks_{self.chunking_strategy}.json"
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
        chunker = TextChunker(
            input_dir=config['cleaned_texts'].get('prefettura_v1.3', 'data/prefettura_v1.3_cleaned_texts'),
            output_dir=config['chunks'].get('prefettura_v1.3', 'data/chunks/prefettura_v1.3_chunks'),
            max_chunk_words=400,
            min_chunk_length=10,
            max_tokens=512,
            chunking_strategy=config.get('chunking_strategy', ChunkingStrategy.PARENT.value),
            tokenizer_name=EncoderModels.ITALIAN_LEGAL_BERT_SC.value
        )
        chunker.process_directory()
        print("Text chunking completed.")
    except Exception as e:
        print(f"Error during text chunking: {e}")