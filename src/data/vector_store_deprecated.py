import json
from typing import Dict, List, Any
from pathlib import Path
import logging
import numpy as np
from rdflib import Graph, Literal, URIRef, Namespace
from neo4j import GraphDatabase
import yaml
from src.utils.logging_utils import setup_logger

class VectorStore:
    """Stores embeddings in GraphDB (RDF) and Neo4j (nodes)."""

    def __init__(
        self,
        input_dir: str = "data/embeddings",
        graphdb_url: str = "http://localhost:7200",
        graphdb_repo: str = "rag_pipeline",
        neo4j_uri: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password"
    ):
        """
        Initialize VectorStore with configuration parameters.

        Args:
            input_dir (str): Directory containing embeddings and metadata.
            graphdb_url (str): GraphDB server URL.
            graphdb_repo (str): GraphDB repository name.
            neo4j_uri (str): Neo4j server URI.
            neo4j_user (str): Neo4j username.
            neo4j_password (str): Neo4j password.
        """
        self.input_dir = Path(input_dir)
        self.graphdb_url = graphdb_url
        self.graphdb_repo = graphdb_repo
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.logger = setup_logger("vector_store")

        # Initialize RDF graph
        self.graph = Graph()
        self.ex = Namespace("http://example.org/")
        self.graph.bind("ex", self.ex)

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.logger.info("Initialized VectorStore with GraphDB and Neo4j connections")

    def close(self):
        """Close Neo4j driver."""
        self.driver.close()

    def load_embedding_metadata(self) -> List[Dict[str, Any]]:
        """
        Load embedding metadata from summary file.

        Returns:
            List[Dict[str, Any]]: List of embedding result dictionaries.
        """
        summary_file = self.input_dir / "embeddings_summary.json"
        if not summary_file.exists():
            self.logger.error("Embeddings summary file not found: %s", summary_file)
            raise FileNotFoundError(f"Embeddings summary file not found: {summary_file}")

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load embeddings summary: %s", str(e))
            raise

    def store_in_graphdb(self, result: Dict[str, Any]) -> None:
        """
        Store embeddings as RDF triples in GraphDB.

        Args:
            result (Dict[str, Any]): Embedding result with metadata.
        """
        try:
            for chunk in result["chunk_embeddings"]:
                if not chunk["is_valid"]:
                    self.logger.warning("Skipping invalid chunk: %s", chunk["chunk_id"])
                    continue

                chunk_uri = URIRef(f"{self.ex}{chunk['chunk_id']}")
                self.graph.add((chunk_uri, self.ex.hasFilePath, Literal(result["file_path"])))
                self.graph.add((chunk_uri, self.ex.chunkId, Literal(chunk["chunk_id"])))
                self.graph.add((chunk_uri, self.ex.wordCount, Literal(chunk["word_count"])))
                self.graph.add((chunk_uri, self.ex.charLength, Literal(chunk["char_length"])))
                self.graph.add((chunk_uri, self.ex.embeddingFile, Literal(chunk["embedding_file"])))

            # Serialize to GraphDB (requires GraphDB REST API)
            self.logger.info("Storing RDF triples for %s in GraphDB", result["file_path"])
            # Note: Actual GraphDB API call depends on rdflib-graphdb or similar
        except Exception as e:
            self.logger.error("Failed to store in GraphDB for %s: %s", result["file_path"], str(e))

    def store_in_neo4j(self, result: Dict[str, Any]) -> None:
        """
        Store embeddings as nodes in Neo4j.

        Args:
            result (Dict[str, Any]): Embedding result with metadata.
        """
        try:
            with self.driver.session() as session:
                for chunk in result["chunk_embeddings"]:
                    if not chunk["is_valid"]:
                        self.logger.warning("Skipping invalid chunk: %s", chunk["chunk_id"])
                        continue

                    embedding_file = self.input_dir / chunk["embedding_file"]
                    embedding = np.load(embedding_file).tolist()

                    query = """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.file_path = $file_path,
                        c.word_count = $word_count,
                        c.char_length = $char_length,
                        c.embedding_file = $embedding_file,
                        c.embedding = $embedding
                    """
                    session.run(query, {
                        "chunk_id": chunk["chunk_id"],
                        "file_path": result["file_path"],
                        "word_count": chunk["word_count"],
                        "char_length": chunk["char_length"],
                        "embedding_file": chunk["embedding_file"],
                        "embedding": embedding
                    })
                self.logger.info("Stored nodes for %s in Neo4j", result["file_path"])
        except Exception as e:
            self.logger.error("Failed to store in Neo4j for %s: %s", result["file_path"], str(e))

    def process_directory(self) -> None:
        """
        Process all embedding metadata and store in databases.
        """
        metadata = self.load_embedding_metadata()
        if not metadata:
            self.logger.warning("No embedding metadata found. Skipping processing.")
            return

        self.logger.info("Processing %d files in %s", len(metadata), self.input_dir)
        processed_files = 0

        for result in metadata:
            if not result["is_valid"]:
                self.logger.warning("Skipping invalid file: %s", result["file_path"])
                continue
            self.store_in_graphdb(result)
            self.store_in_neo4j(result)
            processed_files += 1

        self.logger.info("Processed %d/%d files", processed_files, len(metadata))
        self.close()

if __name__ == "__main__":
    with open('src/configs/config.yaml') as file:
        config = yaml.safe_load(file)
    try:
        store = VectorStore(
            input_dir=config['embeddings']['prefettura_v1'],
            graphdb_url=config['databases']['graphdb']['url'],
            graphdb_repo=config['databases']['graphdb']['repo'],
            neo4j_uri=config['databases']['neo4j']['uri'],
            neo4j_user=config['databases']['neo4j']['user'],
            neo4j_password=config['databases']['neo4j']['password']
        )
        store.process_directory()
        print("Vector storage completed.")
    except Exception as e:
        print(f"Error during vector storage: {e}")