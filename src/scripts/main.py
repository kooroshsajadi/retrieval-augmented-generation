import yaml
import logging
import argparse
import json
import os
from typing import List, Dict, Any
from src.retrieval.retriever import MilvusRetriever
from src.generation.generator import LLMGenerator
from src.utils.logging_utils import setup_logger
from typing import Optional

class RAGOrchestrator:
    """Orchestrates RAG pipeline for query processing and response generation."""

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize RAGOrchestrator.

        Args:
            config_path (str): Path to configuration file.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger(__name__)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize retriever
        retriever_config = self.config.get("retrieval", {})
        self.retriever = MilvusRetriever(
            collection_name=retriever_config.get("collection_name", "gotmat_collection"),
            embedding_model=retriever_config.get("embedding_model", "intfloat/multilingual-e5-large"),
            milvus_host=retriever_config.get("milvus_host", "localhost"),
            milvus_port=retriever_config.get("milvus_port", "19530"),
            logger=self.logger
        )

        # Initialize generator
        generator_config = self.config.get("generation", {})
        self.generator = LLMGenerator(
            model_path=generator_config.get("model_path", "model/opus-mt-it-en"),
            model_type=generator_config.get("model_type", "seq2seq"),
            max_length=generator_config.get("max_length", 128),
            device=generator_config.get("device", "auto"),
            logger=self.logger
        )

        # Output directory
        self.output_dir = self.config.get("output", {}).get("output_dir", "data/destination")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory set to {self.output_dir}")

    def process_query(self, query: str, top_k: int = 5, max_new_tokens: int = 50) -> Dict[str, Any]:
        """
        Process a single query through retrieval and generation.

        Args:
            query (str): User query (in Italian).
            top_k (int): Number of chunks to retrieve.
            max_new_tokens (int): Maximum tokens to generate.

        Returns:
            Dict[str, Any]: Query, retrieved chunks, and response.
        """
        try:
            # Retrieve chunks
            contexts = self.retriever.retrieve(query, top_k=top_k)
            self.logger.info(f"Retrieved {len(contexts)} chunks for query: {query[:50]}...")

            # Generate response
            response = self.generator.generate(query, contexts, max_new_tokens=max_new_tokens)
            self.logger.info(f"Generated response: {response[:100]}...")

            # Save output
            output = {
                "query": query,
                "contexts": contexts,
                "response": response
            }
            output_path = os.path.join(self.output_dir, f"response_{hash(query) % 10000}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved output to {output_path}")

            return output
        except Exception as e:
            self.logger.error(f"Failed to process query '{query}': {str(e)}")
            raise

    def process_batch(self, queries: List[str], top_k: int = 5, max_new_tokens: int = 50) -> List[Dict[str, Any]]:
        """
        Process a batch of queries.

        Args:
            queries (List[str]): List of queries.
            top_k (int): Number of chunks to retrieve.
            max_new_tokens (int): Maximum tokens to generate.

        Returns:
            List[Dict[str, Any]]: Results for each query.
        """
        results = []
        for query in queries:
            result = self.process_query(query, top_k, max_new_tokens)
            results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Orchestrator")
    parser.add_argument("--config", default="configs/rag.yaml", help="Path to config file")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--query-file", help="Path to file with queries (one per line)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    logger = setup_logger(__name__)
    orchestrator = RAGOrchestrator(config_path=args.config, logger=logger)

    if args.query:
        result = orchestrator.process_query(args.query, args.top_k, args.max_new_tokens)
        print(f"Query: {result['query']}\nResponse: {result['response']}")
    elif args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        results = orchestrator.process_batch(queries, args.top_k, args.max_new_tokens)
        for result in results:
            print(f"Query: {result['query']}\nResponse: {result['response']}\n")
    else:
        # Default example query
        query = "Quali sono i requisiti per la residenza in Italia?"
        result = orchestrator.process_query(query, args.top_k, args.max_new_tokens)
        print(f"Query: {result['query']}\nResponse: {result['response']}")

if __name__ == "__main__":
    main()