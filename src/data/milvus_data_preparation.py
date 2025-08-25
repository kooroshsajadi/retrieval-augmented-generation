from pymilvus import model
from pathlib import Path
from typing import List

def read_texts_from_directory(directory_path: str) -> List[str]:
    """
    Reads all .txt files from the given directory and returns a list of their contents.

    Args:
        directory_path (str): Path to the directory containing text files.

    Returns:
        List[str]: List where each element is the text content of one file.
    """
    path = Path(directory_path)
    docs = []
    for file_path in path.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(text)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    return docs

if __name__ == "__main__":
    # If connection to https://huggingface.co/ failed, uncomment the following path
    # import os
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
    embedding_fn = model.DefaultEmbeddingFunction()

    docs = read_texts_from_directory("data/prefettura_v1.2.cleaned_texts")

vectors = embedding_fn.encode_documents(docs)

# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "courthouse"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))