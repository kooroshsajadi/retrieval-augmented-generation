import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from src.utils.logging_utils import setup_logger
from scripts.validate_data import DataValidator
from scripts.ingest_data import DataIngestor
from scripts.sentence_transformer import SentenceTransformerEmbedder
from src.data.vector_store import VectorStore
from src.retrieval.retriever import MilvusRetriever
from src.generation.generator import LLMGenerator
from src.augmentation.augmenter import Augmenter

class RAGOrchestrator:
    """Orchestrates the RAG pipeline for processing user queries and files."""

    def __init__(self, config_path: str = "configs/rag.yaml"):
        """
        Initialize RAGOrchestrator with configuration.

        Args:
            config_path (str): Path to configuration file.
        """
        self.logger = setup_logger("rag_orchestrator")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.validator = DataValidator(
            supported_formats=self.config.get("supported_formats", [".text", ".txt", ".jpg", ".jpeg", ".gif", ".png", ".pdf"]),
            logger=self.logger
        )
        self.data_ingestor = DataIngestor(
            output_dir=self.config["data"]["texts"],
            language="ita",
            tessdata_dir=self.config.get("tessdata_dir", None),
            logger=self.logger
        )
        self.embedder = SentenceTransformerEmbedder(
            model_name=self.config.get("embedding_model", "intfloat/multilingual-e5-large"),
            output_dir=self.config["data"]["embeddings"],
            max_chunk_words=self.config.get("max_chunk_words", 500),
            min_chunk_length=self.config.get("min_chunk_length", 10),
            logger=self.logger
        )
        self.vector_store = VectorStore(
            collection_name=self.config.get("collection_name", "gotmat_collection"),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
            embedding_dim=self.config.get("embedding_dim", 1024),
            chunks_dir=self.config["data"].get("chunks", "data/chunks/prefettura_v1.2_chunks"),
            embeddings_dir=self.config["data"].get("embeddings", "data/embeddings/prefettura_v1.2_embeddings"),
            logger=self.logger
        )
        self.retriever = MilvusRetriever(
            collection_name=self.config.get("collection_name", "gotmat_collection"),
            embedding_model=self.config.get("embedding_model", "intfloat/multilingual-e5-large"),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
            logger=self.logger
        )
        self.augmenter = Augmenter(
            max_contexts=self.config.get("max_contexts", 5),
            max_context_length=self.config.get("max_context_length", 1000),
            logger=self.logger
        )
        self.generator = LLMGenerator(
            model_path=self.config.get("model_path", "models/fine_tuned_models/opus-mt-it-en"),
            adapter_path=self.config.get("adapter_path", None),
            tokenizer_path=self.config.get("tokenizer_path", None),
            model_type="seq2seq",
            max_length=self.config.get("max_length", 128),
            device=self.config.get("device", "auto"),
            logger=self.logger
        )

    def process_file(self, file_path: str) -> bool:
        """
        Process a user-provided file and store its embeddings in Milvus.

        Args:
            file_path (str): Path to the input file.

        Returns:
            bool: True if processing is successful, False otherwise.
        """
        try:
            # Validate file
            validation_result = self.validator.validate_file(file_path)
            if not validation_result["is_valid"]:
                self.logger.error("File validation failed: %s", validation_result["error"])
                return False

            # Extract text
            ingest_result = self.data_ingestor.extract_text(file_path)
            if not ingest_result["is_valid"]:
                self.logger.error("Text extraction failed: %s", ingest_result["error"])
                return False

            # Generate embeddings
            embed_result = self.embedder.process_file(file_path, ingest_result["text"])
            if not embed_result["is_valid"]:
                self.logger.error("Embedding generation failed: %s", embed_result["error"])
                return False

            # Store embeddings in Milvus
            chunk_texts = [c["text"] for c in embed_result["chunk_embeddings"]]
            embeddings = [np.load(Path(self.config["data"]["embeddings"]) / c["embedding_file"]) for c in embed_result["chunk_embeddings"]]
            chunk_ids = [c["chunk_id"] for c in embed_result["chunk_embeddings"]]
            success = self.vector_store.store_vectors(chunk_texts, embeddings, chunk_ids)
            if not success:
                self.logger.error("Failed to store embeddings in Milvus")
                return False

            self.logger.info("Successfully processed and stored embeddings for %s", file_path)
            return True
        except Exception as e:
            self.logger.error("File processing failed for %s: %s", file_path, str(e))
            return False

    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user query and generate a response.

        Args:
            query (str): User query in Italian.
            top_k (int): Number of chunks to retrieve.

        Returns:
            Dict[str, Any]: Dictionary with query, response, and contexts.
        """
        try:
            # Generate query embedding
            query_result = self.embedder.process_query(query)
            if not query_result["is_valid"]:
                self.logger.error("Query embedding failed: %s", query_result["error"])
                return {"query": query, "response": f"Error: {query_result['error']}", "contexts": []}

            # Retrieve relevant chunks
            contexts = self.retriever.retrieve(query, top_k)
            self.logger.info("Retrieved %d contexts for query: %s...", len(contexts), query[:50])

            # Augment query with contexts
            prompt = self.augmenter.augment(query, contexts)

            # Generate response
            response = self.generator.generate(prompt, max_new_tokens=self.config.get("max_new_tokens", 50))
            self.logger.info("Generated response: %s...", response[:100])

            return {"query": query, "response": response, "contexts": contexts}
        except Exception as e:
            self.logger.error("Query processing failed for '%s': %s", query, str(e))
            return {"query": query, "response": f"Error: {str(e)}", "contexts": []}

    def process_queries_from_file(self, queries_file: Union[Path, str],
                                  output_path: Union[Path, str], top_k: int = 5) -> bool:
        """
        Process queries from a JSON file and save results to output JSON.

        Args:
            queries_file Union[Path, str]: Path to JSON file with queries.
            output_path Union[Path, str]: Path to save output JSON.
            top_k (int): Number of chunks to retrieve per query.

        Returns:
            bool: True if processing is successful, False otherwise.
        """
        try:
            # Read queries from JSON
            queries_file = Path(queries_file)
            if not queries_file.exists():
                self.logger.error("Queries file not found: %s", queries_file)
                return False

            with open(queries_file, "r", encoding="utf-8") as f:
                queries_data = json.load(f)

            results = []
            for item in queries_data:
                if "Italian" not in item:
                    self.logger.warning("Skipping item without 'Italian' field: %s", item)
                    continue
                query = item["Italian"]
                result = self.process_query(query, top_k)
                results.append({"query": query, "answer": result["response"]})

            # Save results to JSON
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved query responses to %s", output_path)
            return True
        except Exception as e:
            self.logger.error("Failed to process queries from %s: %s", queries_file, str(e))
            return False

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Orchestrator")
    parser.add_argument("--queries_file", default="data/prompts.json",type=str, help="Path to JSON file with queries")
    parser.add_argument("--file", type=str, help="Path to optional input file (PDF, text, or image)")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, default="data/results/responses.json", help="Path to save query responses")
    args = parser.parse_args()

    orchestrator = RAGOrchestrator(config_path=args.config)

    # Process file if provided
    if args.file:
        success = orchestrator.process_file(args.file)
        if not success:
            print(f"Failed to process file: {args.file}")
            return

    # Process queries from file if provided
    if args.queries_file:
        success = orchestrator.process_queries_from_file(args.queries_file, args.output)
        if success:
            print(f"Successfully processed queries from {args.queries_file} and saved to {args.output}")
        else:
            print(f"Failed to process queries from {args.queries_file}")
    else:
        print("No queries file provided. Use --queries_file to specify a JSON file with queries.")

if __name__ == "__main__":
    main()