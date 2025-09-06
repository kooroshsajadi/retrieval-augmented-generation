from pathlib import Path
from typing import List, Optional
import numpy as np
import logging
import argparse
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
        self.collection = self._create_collection(force_recreate=False)

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

    def _read_chunk_file_names(self) -> List[str]:
        """
        Read all .txt file stems in the chunks directory to get valid chunk IDs.
    
        Returns:
            List[str]: Sorted list of chunk ID strings (filename stems).
        """
        chunk_file_names = sorted([file_path.stem for file_path in self.chunks_dir.glob("*.txt")])
        self.logger.info("Found %d chunk text files in %s", len(chunk_file_names), self.chunks_dir)
        return chunk_file_names

    def _create_collection(self, force_recreate: bool = False) -> Collection:
        """
        Create or use existing Milvus collection with the defined schema.

        Args:
            force_recreate (bool): If True, drop and recreate the collection; if False, use existing collection or create if absent.

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

        # Drop collection if force_recreate is True
        if force_recreate and has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.logger.info("Dropped existing collection: %s", self.collection_name)

        # Create collection if it doesn't exist
        if not has_collection(self.collection_name):
            collection = Collection(name=self.collection_name, schema=schema)
            self.logger.info("Created collection: %s", self.collection_name)
        else:
            collection = Collection(name=self.collection_name)
            self.logger.info("Using existing collection: %s", self.collection_name)

        return collection

    def store_vectors(self, texts: List[str], embeddings: List[np.ndarray], chunk_ids: Optional[List[str]] = None, subject: str = "courthouse", force_recreate: bool = False) -> bool:
        """
        Store embeddings and associated metadata in Milvus.

        Args:
            texts (List[str]): List of chunk texts.
            embeddings (List[np.ndarray]): List of embedding vectors.
            chunk_ids (Optional[List[str]]): List of chunk IDs, defaults to None (auto-generated).
            subject (str): Subject metadata for all chunks (default: 'courthouse').
            force_recreate (bool): If True, recreate the collection before insertion; if False, append to existing collection.

        Returns:
            bool: True if insertion and indexing succeed, False otherwise.
        """
        try:
            # Recreate collection if force_recreate is True
            if force_recreate:
                self.collection = self._create_collection(force_recreate=True)

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
            ids, embeddings, chunk_ids, texts, subjects = map(list, zip(*valid_entities))

            # Insert into Milvus
            entities = [list(ids), list(embeddings), list(chunk_ids), list(texts), list(subjects)]
            insertion_result = self.collection.insert(entities)
            self.collection.flush()
            self.logger.info("Inserted %d entities into collection %s", len(insertion_result.primary_keys), self.collection_name)

            # Create index (only if no index exists)
            if not self.collection.has_index():
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

    def bulk_insert(self, texts_dir: Optional[str] = None) -> bool:
        """
        Perform bulk insertion of all chunk texts and embeddings from directories.

        Args:
            texts_dir (Optional[str]): Directory containing original text files, defaults to None.

        Returns:
            bool: True if bulk insertion succeeds, False otherwise.
        """
        try:
            # Ensure directories exist
            for dir_path in [self.chunks_dir, self.embeddings_dir]:
                if not dir_path.exists():
                    self.logger.error("Directory not found: %s", dir_path)
                    return False

            # Get chunk IDs from chunk files
            chunk_ids = self._read_chunk_file_names()
            if not chunk_ids:
                self.logger.error("No chunk files found in %s", self.chunks_dir)
                return False

            # Load texts and embeddings
            texts = []
            embeddings = []
            valid_chunk_ids = []
            for chunk_id in chunk_ids:
                # Load chunk text
                text = self._load_chunk_text(chunk_id)
                if not text:
                    self.logger.warning("Skipping chunk %s due to empty or missing text", chunk_id)
                    continue

                # Load embedding
                embedding_file = self.embeddings_dir / f"{chunk_id}.npy"
                try:
                    embedding = np.load(embedding_file)
                    if embedding.shape[0] != self.embedding_dim:
                        self.logger.warning("Embedding for chunk %s has unexpected dimension %s, skipping", chunk_id, embedding.shape)
                        continue
                except FileNotFoundError:
                    self.logger.warning("Embedding file %s not found, skipping", embedding_file)
                    continue

                texts.append(text)
                embeddings.append(embedding)
                valid_chunk_ids.append(chunk_id)

            if not texts:
                self.logger.error("No valid texts or embeddings to insert")
                return False

            # Store data in Milvus with force_recreate=True
            success = self.store_vectors(
                texts=texts,
                embeddings=embeddings,
                chunk_ids=valid_chunk_ids,
                subject="courthouse",
                force_recreate=True
            )
            if success:
                self.logger.info("Bulk insertion completed successfully for %d chunks", len(valid_chunk_ids))
            else:
                self.logger.error("Bulk insertion failed")
            return success
        except Exception as e:
            self.logger.error("Bulk insertion failed: %s", str(e))
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk insert data into Milvus collection")
    parser.add_argument("--collection_name", type=str, default="gotmat_collection", help="Milvus collection name")
    parser.add_argument("--milvus_host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--milvus_port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Dimension of embedding vectors")
    parser.add_argument("--chunks_dir", type=str, default="data/chunks/prefettura_v1.2_chunks", help="Directory containing chunked text files")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings/prefettura_v1.2_embeddings", help="Directory containing embedding files")
    parser.add_argument("--texts_dir", type=str, default=None, help="Directory containing original text files (optional)")
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger("vector_store_main")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # Initialize VectorStore
    vector_store = VectorStore(
        collection_name=args.collection_name,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        embedding_dim=args.embedding_dim,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        logger=logger
    )

    # Perform bulk insertion
    success = vector_store.bulk_insert(texts_dir=args.texts_dir)
    if success:
        logger.info("Bulk insertion completed successfully")
    else:
        logger.error("Bulk insertion failed")