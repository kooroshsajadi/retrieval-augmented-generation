from pathlib import Path
from typing import List
import numpy as np
from pymilvus import (
    connections, has_collection,
    FieldSchema, CollectionSchema, DataType, Collection
)

def load_chunk_text(chunk_id: str, chunks_dir: Path) -> str:
    chunk_file = chunks_dir / f"{chunk_id}.txt"
    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: chunk text file {chunk_file} not found.")
        return ""

def read_chunk_file_names(directory_path: str) -> set:
    """
    Reads all .txt files in the given directory and returns a set of their filename stems 
    (i.e., filenames without extensions). This is intended to collect valid chunk IDs 
    for comparison.

    Args:
        directory_path (str): Path to the directory containing chunk text files.

    Returns:
        set: Set of filename stems representing chunk IDs.
    """
    path = Path(directory_path)
    chunk_file_names = set()
    for file_path in path.glob("*.txt"):
        chunk_file_names.add(file_path.stem)  # filename without '.txt'
    return chunk_file_names

def main():
    # Connect to Milvus standalone server running on localhost
    connections.connect(alias="default", host="localhost", port="19530")

    collection_name = "gotmat_collection"

    # Define schema with primary key, vector field, and metadata
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
    ]
    schema = CollectionSchema(fields=fields, description="Collection of embeddings with metadata")

    # Drop collection if exists
    if has_collection(collection_name):
        Collection(collection_name).drop()

    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection: {collection_name}")

    # Load all chunk texts
    chunk_names = read_chunk_file_names("data/chunks/prefettura_v1.2_chunks")
    print(f"Loaded {len(chunk_names)} text chunks names.")

    # Directory and list of embedding files
    embedding_dir = Path("data/embeddings/prefettura_v1.2_embeddings")
    embedding_files = sorted(embedding_dir.glob("*.npy"))
    print(f"Found {len(embedding_files)} embedding files.")

    # Prepare data lists for insertion
    ids = []
    embeddings = []
    chunk_ids = []
    texts = []
    subjects = []

    for i, emb_file in enumerate(embedding_files):
        chunk_id = emb_file.stem

        if chunk_id not in chunk_names:
            print(f"Warning: No text file found for chunk {chunk_id}, skipping.")
            continue

        embedding_vector = np.load(emb_file)
        if embedding_vector.shape[0] != 1024:
            print(f"Warning: Embedding for {chunk_id} has unexpected dimension {embedding_vector.shape}, skipping.")
            continue

        ids.append(i)
        embeddings.append(embedding_vector.tolist())  # Convert numpy array to list
        chunk_ids.append(chunk_id)
        texts.append(load_chunk_text(chunk_id, Path("data/chunks/prefettura_v1.2_chunks")))
        subjects.append("courthouse")  # or your subject metadata

    if not ids:
        print("No data to insert, exiting.")
        return

    print(f"Prepared {len(ids)} entities for insertion.")

    # Insert into Milvus (field-wise lists)
    entities = [ids, embeddings, chunk_ids, texts, subjects]
    insertion_result = collection.insert(entities)
    collection.flush()  # Ensure data persisted

    print(f"Inserted {len(insertion_result.primary_keys)} entities into collection {collection_name}.")

    # Create index on vector field - REQUIRED before loading and searching
    index_params = {
        "index_type": "IVF_FLAT",  # Choose suitable index type; e.g., IVF_FLAT, HNSW, or ANNOY
        "metric_type": "L2",       # Must match your search metric
        "params": {"nlist": 128}   # Tune nlist according to dataset size and recall requirements
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created on 'embedding' field.")

    # Load collection to make vectors and index ready for search
    collection.load()
    print("Collection loaded and ready for search.")

if __name__ == "__main__":
    main()
