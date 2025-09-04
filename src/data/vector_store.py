from pathlib import Path
from typing import List, Optional
import numpy as np
import logging
from pymilvus import (
    connections, has_collection,
    FieldSchema, CollectionSchema, DataType, Collection
)
from src.utils.logging_utils import setup_logger

class VectorStore:
    """Manages storage and indexing of embeddings in Milvus for the RAG pipeline."""

    def __init__(
        self,
        collection_name: str = "gotmat_collection",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        embedding_dim: int = 1024,
        chunks_dir: str = "data/chunks/prefettura_v1.2_chunks",
        embeddings_dir: str = "data/embeddings/prefettura_v1.2_embeddings",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize VectorStore with Milvus connection and collection settings.

        Args:
            collection_name (str): Name of the Milvus collection.
            milvus_host (str): Milvus server host.
            milvus_port (str): Milvus server port.
            embedding_dim (int): Dimension of embedding vectors.
            chunks_dir (str): Directory containing chunked text files.
            embeddings_dir (str): Directory containing embedding files (.npy).
            logger (Optional[logging.Logger]): Logger instance, defaults to None.
        """
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        self.chunks_dir = Path(chunks_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.logger = logger or setup_logger("src.data.vector_store")
        
        # Connect to Milvus
        try:
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            self.logger.info("Connected to Milvus at %s:%s", self.milvus_host, self.milvus_port)
        except Exception as e:
            self.logger.error("Failed to connect to Milvus: %s", str(e))
            raise

        # Initialize collection
        self.collection = self._create_collection()

    def _load_chunk_text(self, chunk_id: str) -> str:
        """
        Load text for a given chunk ID from the chunks directory.

        Args:
            chunk_id (str): ID of the chunk (filename stem).

        Returns:
            str: Chunk text, or empty string if not found.
        """
        chunk_file = self.chunks_dir / f"{chunk_id}.txt"
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                self.logger.debug("Loaded chunk text from %s", chunk_file)
                return f.read()
        except FileNotFoundError:
            self.logger.warning("Chunk text file %s not found", chunk_file)
            return ""

    def _read_chunk_file_names(self) -> set:
        """
        Read all .txt file stems in the chunks directory to get valid chunk IDs.

        Returns:
            set: Set of chunk ID strings (filename stems).
        """
        chunk_file_names = {file_path.stem for file_path in self.chunks_dir.glob("*.txt")}
        self.logger.info("Found %d chunk text files in %s", len(chunk_file_names), self.chunks_dir)
        return chunk_file_names

    def _create_collection(self) -> Collection:
        """
        Create or recreate the Milvus collection with the defined schema.

        Returns:
            Collection: Milvus collection object.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields=fields, description="Collection of embeddings with metadata")

        # Drop collection if it exists
        if has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.logger.info("Dropped existing collection: %s", self.collection_name)

        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)
        self.logger.info("Created collection: %s", self.collection_name)
        return collection

    def store_vectors(self, texts: List[str], embeddings: List[np.ndarray], chunk_ids: Optional[List[str]] = None, subject: str = "courthouse") -> bool:
        """
        Store embeddings and associated metadata in Milvus.

        Args:
            texts (List[str]): List of chunk texts.
            embeddings (List[np.ndarray]): List of embedding vectors.
            chunk_ids (Optional[List[str]]): List of chunk IDs, defaults to None (auto-generated).
            subject (str): Subject metadata for all chunks (default: 'courthouse').

        Returns:
            bool: True if insertion and indexing succeed, False otherwise.
        """
        try:
            # Validate inputs
            if len(texts) != len(embeddings):
                self.logger.error("Mismatch between number of texts (%d) and embeddings (%d)", len(texts), len(embeddings))
                return False
            if chunk_ids and len(chunk_ids) != len(texts):
                self.logger.error("Mismatch between number of chunk IDs (%d) and texts (%d)", len(chunk_ids), len(texts))
                return False

            # Prepare data for insertion
            ids = list(range(len(texts)))
            if chunk_ids is None:
                chunk_ids = [f"chunk_{i}" for i in ids]
            subjects = [subject] * len(texts)
            valid_entities = []
            for i, (text, embedding, chunk_id) in enumerate(zip(texts, embeddings, chunk_ids)):
                if embedding.shape[0] != self.embedding_dim:
                    self.logger.warning("Embedding for chunk %s has unexpected dimension %s, skipping", chunk_id, embedding.shape)
                    continue
                valid_entities.append((i, embedding.tolist(), chunk_id, text, subject))

            if not valid_entities:
                self.logger.error("No valid entities to insert")
                return False

            # Unzip valid entities
            ids, embeddings, chunk_ids, texts, subjects = zip(*valid_entities)

            # Insert into Milvus
            entities = [list(ids), list(embeddings), list(chunk_ids), list(texts), list(subjects)]
            insertion_result = self.collection.insert(entities)
            self.collection.flush()
            self.logger.info("Inserted %d entities into collection %s", len(insertion_result.primary_keys), self.collection_name)

            # Create index
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.logger.info("Index created on 'embedding' field")

            # Load collection for search
            self.collection.load()
            self.logger.info("Collection %s loaded and ready for search", self.collection_name)
            return True
        except Exception as e:
            self.logger.error("Failed to store vectors: %s", str(e))
            return False

# if __name__ == "__main__":
#     # Example usage
#     vector_store = VectorStore(
#         collection_name="gotmat_collection",
#         milvus_host="localhost",
#         milvus_port="19530",
#         embedding_dim=1024,
#         chunks_dir="data/chunks/prefettura_v1.2_chunks",
#         embeddings_dir="data/embeddings/prefettura_v1.2_embeddings"
#     )
#     # Example data
#     texts = ["Sample text 1", "Sample text 2"]
#     embeddings = [np.random.rand(1024), np.random.rand(1024)]
#     chunk_ids = ["sample_chunk_1", "sample_chunk_2"]
#     vector_store.store_vectors(texts, embeddings, chunk_ids)