from enum import Enum

class ChunkingStrategy(Enum):
    """Enum for chunking strategies used in TextChunker."""
    SENTENCE_BASED = "sentence-based"
    PARENT = "parent"