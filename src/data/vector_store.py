from pathlib import Path
from typing import List, Optional
import numpy as np
import logging
import json
import argparse
from pymilvus import (
    connections, has_collection,
    FieldSchema, CollectionSchema, DataType, Collection, utility
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
        metadata: Optional[List[dict]] = None,
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
        self.metadata = []
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
        self.collection = self._create_collection(force_recreate=False)

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
        Create or use existing Milvus collection with the defined schema and ensure index exists.

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
            FieldSchema(name="parent_file_path", dtype=DataType.VARCHAR, max_length=500)
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

        # Ensure index exists on the embedding field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        indexes = collection.indexes
        if not any(index.field_name == "embedding" for index in indexes):
            collection.create_index(field_name="embedding", index_params=index_params)
            self.logger.info("Created index on embedding field for collection: %s", self.collection_name)
        else:
            self.logger.info("Index on embedding field already exists for collection: %s", self.collection_name)

        return collection

    def store_vectors(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        chunk_ids: List[str],
        parent_ids: List[str],
        parent_file_paths: List[str],
        subject: str = "courthouse"
    ) -> bool:
        """
        Store vectors and metadata in Milvus with batch processing.

        Args:
            texts (List[str]): List of chunk texts.
            embeddings (List[np.ndarray]): List of embedding vectors.
            chunk_ids (List[str]): List of chunk IDs.
            parent_ids (List[str]): List of parent IDs.
            parent_file_paths (List[str]): List of parent file paths.
            subject (str): Subject of the chunks (e.g., 'courthouse').

        Returns:
            bool: True if storage is successful, False otherwise.
        """
        try:
            if not (len(texts) == len(embeddings) == len(chunk_ids) == len(parent_ids) == len(parent_file_paths)):
                self.logger.error("Input lists have different lengths")
                return False

            batch_size = 100  # Adjust based on testing
            total = len(texts)
            self.logger.info("Storing %d vectors in Milvus with batch size %d", total, batch_size)

            # Get the current maximum ID to avoid conflicts
            self.collection.load()
            max_id = 0
            try:
                result = self.collection.query(expr="id >= 0", output_fields=["id"], limit=1, sort_field="id", sort_order="desc")
                if result:
                    max_id = result[0]["id"]
            except Exception as e:
                self.logger.warning("Could not retrieve max ID, starting from 1: %s", str(e))

            for i in range(0, total, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = [emb.tolist() for emb in embeddings[i:i + batch_size]]
                batch_chunk_ids = chunk_ids[i:i + batch_size]
                batch_parent_ids = parent_ids[i:i + batch_size]
                batch_parent_file_paths = parent_file_paths[i:i + batch_size]
                batch_subjects = [subject] * len(batch_texts)
                # Generate sequential IDs starting from max_id + 1
                batch_ids = list(range(max_id + 1, max_id + 1 + len(batch_texts)))
                max_id += len(batch_texts)

                # Insert data as a list of field values, including explicit 'id'
                data = [
                    batch_ids,               # id
                    batch_embeddings,        # embedding
                    batch_chunk_ids,         # chunk_id
                    batch_texts,             # text
                    batch_subjects,          # subject
                    batch_parent_ids,        # parent_id
                    batch_parent_file_paths  # parent_file_path
                ]
                self.collection.insert(data)
                self.logger.debug("Inserted batch %d-%d of %d", i, min(i + batch_size, total), total)

            self.collection.flush()
            self.logger.info("Successfully stored %d vectors in Milvus", total)

            # Update metadata
            for chunk_id, parent_id, parent_file_path in zip(chunk_ids, parent_ids, parent_file_paths):
                self.metadata.append({
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "parent_file_path": str(parent_file_path)
                })
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            self.logger.info("Updated metadata at %s", self.metadata_path)
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
            # Initialize collection
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
                parent_file_paths=parent_file_paths
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