from pathlib import Path
from typing import List, Optional
import numpy as np
import logging
import json
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
        embedding_dim: int = 768,  # Matches dlicari/Italian-Legal-BERT-SC
        chunks_dir: str = "data/chunks/prefettura_v1.3_chunks",
        embeddings_dir: str = "data/embeddings/prefettura_v1.3_embeddings",
        metadata_path: str = "data/embeddings/prefettura_v1.3_embeddings/embeddings_prefettura_v1.3.json",
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
            metadata_path (str): Path to embedding metadata file.
            logger (Optional[logging.Logger]): Logger instance, defaults to None.
        """
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        self.chunks_dir = Path(chunks_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_path = Path(metadata_path)
        self.logger = logger or setup_logger("src.data.vector_store")
        
        # Validate directories
        for dir_path in [self.chunks_dir, self.embeddings_dir]:
            if not dir_path.exists():
                self.logger.error("Directory not found: %s", dir_path)
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Connect to Milvus
        try:
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            self.logger.info("Connected to Milvus at %s:%s", self.milvus_host, self.milvus_port)
        except Exception as e:
            self.logger.error("Failed to connect to Milvus: %s", str(e))
            raise

        # Initialize collection
        # self.collection = self._create_collection(force_recreate=False)

    def _load_chunk_text(self, file_path: str) -> str:
        """
        Load text for a given chunk file path.

        Args:
            file_path (str): Path to the chunk file.

        Returns:
            str: Chunk text, or empty string if not found.
        """
        chunk_file = Path(file_path)
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                self.logger.debug("Loaded chunk text from %s", chunk_file)
                return f.read()
        except FileNotFoundError:
            self.logger.warning("Chunk text file %s not found", chunk_file)
            return ""

    def _load_embedding_metadata(self) -> List[dict]:
        """
        Load embedding metadata from the metadata file.

        Returns:
            List[dict]: List of file metadata dictionaries.
        """
        if not self.metadata_path.exists():
            self.logger.error("Embedding metadata file not found: %s", self.metadata_path)
            raise FileNotFoundError(f"Embedding metadata file not found: {self.metadata_path}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.logger.info("Loaded embedding metadata from %s", self.metadata_path)
            return metadata
        except Exception as e:
            self.logger.error("Failed to load embedding metadata: %s", str(e))
            raise

    def _create_collection(self, force_recreate: bool = False) -> Collection:
        """
        Create or use existing Milvus collection with the defined schema.

        Args:
            force_recreate (bool): If True, drop and recreate the collection; if False, use existing or create.

        Returns:
            Collection: Milvus collection object.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="parent_file_path", dtype=DataType.VARCHAR, max_length=500)  # Added for parent chunk access
        ]
        schema = CollectionSchema(fields=fields, description="Collection of embeddings with metadata")

        if force_recreate and has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.logger.info("Dropped existing collection: %s", self.collection_name)

        if not has_collection(self.collection_name):
            collection = Collection(name=self.collection_name, schema=schema)
            self.logger.info("Created collection: %s", self.collection_name)
        else:
            collection = Collection(name=self.collection_name)
            self.logger.info("Using existing collection: %s", self.collection_name)

        return collection

    def store_vectors(self, texts: List[str],
                      embeddings: List[np.ndarray],
                      chunk_ids: List[str],
                      subject: str = "courthouse",
                      parent_ids: Optional[List[str]] = None,
                      parent_file_paths: Optional[List[str]] = None,
                      force_recreate: bool = False) -> bool:
        """
        Store embeddings and associated metadata in Milvus.

        Args:
            texts (List[str]): List of chunk texts.
            embeddings (List[np.ndarray]): List of embedding vectors.
            chunk_ids (List[str]): List of chunk IDs.
            subject (str): Subject metadata for all chunks (default: 'courthouse').
            parent_ids (List[str]): List of parent IDs for parent chunking.
            parent_file_paths (List[str]): List of parent file paths for parent chunk access.
            force_recreate (bool): If True, recreate the collection before insertion.

        Returns:
            bool: True if insertion and indexing succeed, False otherwise.
        """
        try:
            if force_recreate:
                self.collection = self._create_collection(force_recreate=True)

            if len(texts) != len(embeddings) or len(texts) != len(chunk_ids):
                self.logger.error("Mismatch: texts (%d), embeddings (%d), chunk_ids (%d)", 
                                 len(texts), len(embeddings), len(chunk_ids))
                return False
            if parent_ids and len(parent_ids) != len(texts):
                self.logger.error("Mismatch: parent_ids (%d), texts (%d)", len(parent_ids), len(texts))
                return False
            if parent_file_paths and len(parent_file_paths) != len(texts):
                self.logger.error("Mismatch: parent_file_paths (%d), texts (%d)", len(parent_file_paths), len(texts))
                return False

            parent_ids = parent_ids or [None] * len(texts)
            parent_file_paths = parent_file_paths or [None] * len(texts)
            subjects = [subject] * len(texts)
            valid_entities = []
            for i, (text, embedding, chunk_id, parent_id, parent_file_path) in enumerate(
                zip(texts, embeddings, chunk_ids, parent_ids, parent_file_paths)
            ):
                if embedding.shape[0] != self.embedding_dim:
                    self.logger.warning("Embedding for chunk %s has dimension %s, expected %d, skipping", 
                                      chunk_id, embedding.shape, self.embedding_dim)
                    continue
                valid_entities.append((i, embedding.tolist(), chunk_id, text, subject, parent_id, parent_file_path))

            if not valid_entities:
                self.logger.error("No valid entities to insert")
                return False

            ids, embeddings, chunk_ids, texts, subjects, parent_ids, parent_file_paths = map(list, zip(*valid_entities))

            entities = [ids, embeddings, chunk_ids, texts, subjects, parent_ids, parent_file_paths]
            insertion_result = self.collection.insert(entities)
            self.collection.flush()
            self.logger.info("Inserted %d entities into collection %s", len(insertion_result.primary_keys), self.collection_name)

            if not self.collection.has_index():
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                self.logger.info("Index created on 'embedding' field")

            self.collection.load()
            self.logger.info("Collection %s loaded and ready for search", self.collection_name)
            return True
        except Exception as e:
            self.logger.error("Failed to store vectors: %s", str(e))
            return False

    def bulk_insert(self, texts_dir: Optional[str] = None) -> bool:
        """
        Perform bulk insertion of all chunk texts and embeddings using metadata.

        Args:
            texts_dir (Optional[str]): Directory containing original text files, defaults to None.

        Returns:
            bool: True if bulk insertion succeeds, False otherwise.
        """
        try:
            # Rrecreate collection
            # if has_collection(self.collection_name):
            #     Collection(self.collection_name).drop()
            #     self.logger.info("Dropped existing collection: %s", self.collection_name)
            self.collection = self._create_collection(force_recreate=False)

            # Load embedding metadata
            metadata = self._load_embedding_metadata()
            if not metadata:
                self.logger.error("No embedding metadata found")
                return False

            texts = []
            embeddings = []
            chunk_ids = []
            parent_ids = []
            parent_file_paths = []

            for file_metadata in metadata:
                if not file_metadata["is_valid"]:
                    self.logger.warning("Skipping invalid file: %s", file_metadata["file_path"])
                    continue

                for chunk_meta in file_metadata["chunk_embeddings"]:
                    if not chunk_meta["is_valid"]:
                        self.logger.warning("Skipping invalid chunk: %s", chunk_meta["chunk_id"])
                        continue

                    # Load chunk text
                    text = self._load_chunk_text(chunk_meta["file_path"])
                    if not text:
                        self.logger.warning("Skipping chunk %s due to empty or missing text", chunk_meta["chunk_id"])
                        continue

                    # Load embedding
                    embedding_file = self.embeddings_dir / chunk_meta["embedding_file"]
                    try:
                        embedding = np.load(embedding_file)
                        if embedding.shape[0] != self.embedding_dim:
                            self.logger.warning("Embedding for chunk %s has dimension %s, expected %d, skipping", 
                                              chunk_meta["chunk_id"], embedding.shape, self.embedding_dim)
                            continue
                    except FileNotFoundError:
                        self.logger.warning("Embedding file %s not found, skipping", embedding_file)
                        continue

                    texts.append(text)
                    embeddings.append(embedding)
                    chunk_ids.append(chunk_meta["chunk_id"])
                    parent_ids.append(chunk_meta["parent_id"])
                    parent_file_paths.append(chunk_meta["parent_file_path"])

            total_triplets = len(chunk_ids)
            if not total_triplets:
                self.logger.error("No valid text-embedding pairs to insert")
                return False

            success = self.store_vectors(
                texts=texts,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
                subject="courthouse",
                parent_ids=parent_ids,
                parent_file_paths=parent_file_paths,
                force_recreate=False
            )
            if success:
                self.logger.info("Bulk insertion completed: %d triplets inserted", total_triplets)
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
    parser.add_argument("--embedding_dim", type=int, default=768, help="Dimension of embedding vectors")
    parser.add_argument("--chunks_dir", type=str, default="data/chunks/prefettura_v1.3_chunks", help="Directory containing chunked text files")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings/prefettura_v1.3_embeddings", help="Directory containing embedding files")
    parser.add_argument("--metadata_path", type=str, default="data/embeddings/prefettura_v1.3_embeddings/embeddings_prefettura_v1.3.json", help="Embedding metadata file")
    args = parser.parse_args()

    logger = setup_logger("src.data.vector_store")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    vector_store = VectorStore(
        collection_name=args.collection_name,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        embedding_dim=args.embedding_dim,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        metadata_path=args.metadata_path,
        logger=logger
    )

    success = vector_store.bulk_insert()
    if success:
        logger.info("Bulk insertion completed successfully")
    else:
        logger.error("Bulk insertion failed")