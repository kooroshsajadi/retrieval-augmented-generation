import argparse
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from src.utils.logging_utils import setup_logger
from src.validation.validate_data import DataValidator
from src.ingestion.ingest_data import DataIngestor
from src.embeddings.sentence_transformer import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.retrieval.retriever import MilvusRetriever
from src.generation.generator import LLMGenerator
from src.augmentation.augmenter import Augmenter
from src.utils.models.bi_encoders import EncoderModels
from src.utils.models.llms import LargeLanguageModels
from src.utils.models.model_types import ModelTypes

class RAGOrchestrator:
    """Orchestrates the RAG pipeline for processing user queries and files."""

    def __init__(self, config_path: str = "configs/rag.yaml", extended: bool = False):
        """
        Initialize RAGOrchestrator with configuration.

        Args:
            config_path (str): Path to configuration file.
            extended (bool): If True, include extended output with contexts.
        """
        self.logger = setup_logger("scripts.rag_orchestrator")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error("Failed to load config %s: %s", config_path, str(e))
            raise
        self.extended = extended

        # Initialize components
        self.validator = DataValidator(
            supported_formats=self.config.get("supported_formats", [".text", ".txt", ".pdf"]),
            logger=self.logger
        )

        self.data_ingestor = DataIngestor(
            output_dir=self.config["data"]["texts"],
            language="ita",
            tessdata_dir=self.config.get("tessdata_dir", None),
            logger=self.logger
        )

        self.embedder = EmbeddingGenerator(
            model_name=self.config["model"].get("embedding_model", EncoderModels.ITALIAN_LEGAL_BERT_SC.value),
            output_dir=self.config["data"]["embeddings"],
            max_chunk_length=self.config.get("max_chunk_length", 2000),
            min_chunk_length=self.config.get("min_chunk_length", 10),
            logger=self.logger
        )

        self.vector_store = VectorStore(
            collection_name=self.config.get("collection_name", "gotmat_collection"),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
            embedding_dim=self.config.get("embedding_dim", 768),
            chunks_dir=self.config["data"].get("chunks", "data/chunks/prefettura_v1.3.1_chunks"),
            embeddings_dir=self.config["data"].get("embeddings", "data/embeddings/prefettura_v1.3.1_embeddings"),
            metadata_path=self.config["data"].get("embeddings_metadata", "data/embeddings/prefettura_v1.3.1_embeddings/embeddings_prefettura_v1.3.1.json"),
            logger=self.logger
        )

        self.retriever = MilvusRetriever(
            collection_name=self.config.get("collection_name", "gotmat_collection"),
            embedding_model=self.config["model"].get("embedding_model", EncoderModels.ITALIAN_LEGAL_BERT_SC.value),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
            reranker_model=self.config["model"].get("reranker_model", EncoderModels.ITALIAN_LEGAL_BERT.value),
            logger=self.logger
        )

        self.augmenter = Augmenter(
            max_contexts=self.config.get("max_augmentation_contexts", 5),
            max_context_length=self.config.get("max_context_length", 1000),
            max_parent_length=self.config.get("max_parent_length", 2000),
            # logger=self.logger
        )
        
        self.generator = LLMGenerator(
            model_path=self.config['model'].get("model_path", LargeLanguageModels.MBART_LARGE_50.value),
            adapter_path=self.config['model'].get("adapter_path", None),
            tokenizer_path=self.config['model'].get("tokenizer_path", None),
            model_type=self.config['model'].get("model_type", ModelTypes.CASUAL.value),
            max_length=self.config.get("max_input_tokenization_length", 2048),
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
            parent_ids = [c.get("parent_id") for c in embed_result["chunk_embeddings"]]
            parent_file_paths = [c.get("parent_file_path") for c in embed_result["chunk_embeddings"]]
            success = self.vector_store.store_vectors(
                texts=chunk_texts,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
                parent_ids=parent_ids,
                parent_file_paths=parent_file_paths,
                subject="courthouse"
            )
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
            # Retrieve relevant chunks
            contexts = self.retriever.retrieve(query, top_k)
            self.logger.info("Retrieved %d contexts for query: %s...", len(contexts), query[:50])

            # Augment query with contexts
            prompt = self.augmenter.augment(query, contexts)

            # Generate response
            response = self.generator.generate(prompt, max_new_tokens=self.config.get("max_new_tokens", 200))
            self.logger.info("Generated response: %s...", response[:100])

            return {"query": query, "response": response, "contexts": contexts, "prompt": prompt}
        except Exception as e:
            self.logger.error("Query processing failed for '%s': %s", query, str(e))
            return {"query": query, "response": f"Error: {str(e)}", "contexts": [], "prompt": ""}

    def process_queries_from_file(
        self,
        queries_file: Union[Path, str],
        output_path: Union[Path, str],
        top_k: int = 5,
        extended: bool = False
    ) -> bool:
        """
        Process queries from a JSON file and save results to output JSON.

        Args:
            queries_file (Union[Path, str]): Path to JSON file with queries.
            output_path (Union[Path, str]): Path to save output JSON.
            top_k (int): Number of chunks to retrieve per query.
            extended (bool): If True, include top-k chunks in output JSON and print to console.

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
            prompts = []
            for item in queries_data:
                if "Italian" not in item:
                    self.logger.warning("Skipping item without 'Italian' field: %s", item)
                    continue
                query = item["Italian"]
                result = self.process_query(query, top_k)
                output_item = {
                    "query": query,
                    "answer": result["response"]
                }
                if extended: # Include prompt and contexts in output JSON if extended is True
                    output_item["prompt"] = result["prompt"]
                    output_item["contexts"] = [
                        {
                            "chunk_id": context["chunk_id"],
                            "text": context["text"],
                            "score": context["score"],
                            "parent_id": context.get("parent_id"),
                            "parent_text": context.get("parent_text")
                        } for context in result["contexts"]
                    ]

                results.append(output_item)
                prompts.append(result["prompt"])

                # Print extended output to console if requested
                if extended:
                    self.logger.info("Query: %s", query)
                    self.logger.info("Answer: %s", result["response"])
                    self.logger.info("Top-%d closest chunks:", top_k)
                    for i, context in enumerate(result["contexts"], 1):
                        self.logger.info("Chunk %d:", i)
                        self.logger.info("  Chunk ID: %s", context["chunk_id"])
                        self.logger.info("  Text: %s...", context["text"][:100])
                        self.logger.info("  Score: %.4f", context["score"])
                        self.logger.info("  Parent ID: %s", context.get("parent_id", "None"))
                        self.logger.info("  Parent Text: %s...", context.get("parent_text", "None")[:100])
                    self.logger.info("-" * 50)

            # Save results to JSON TODO: Add to a function
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            with open('data/results/prompts_v1.3.1.json', "w", encoding="utf-8") as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved query responses to %s", output_path)
            return True
        except Exception as e:
            self.logger.error("Failed to process queries from %s: %s", queries_file, str(e))
            return False

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Orchestrator")
    parser.add_argument("--queries_file", default="data/prompts.json", type=str, help="Path to JSON file with queries")
    parser.add_argument("--file", type=str, help="Path to optional input file (PDF, text)")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, default="data/results/responses_(leggi_area3)(reranking_bm25_deduplication)_falcon7b_extended.json", help="Path to save query responses")
    parser.add_argument("--extended", action="store_true", help="Print extended output with top-k closest chunks")
    args = parser.parse_args()

    orchestrator = RAGOrchestrator(config_path=args.config, extended=args.extended)

    # Process file if provided
    if args.file:
        success = orchestrator.process_file(args.file)
        if not success:
            print(f"Failed to process file: {args.file}")
            return

    # Process queries from file if provided
    if args.queries_file:
        success = orchestrator.process_queries_from_file(
            queries_file=args.queries_file,
            output_path=args.output,
            top_k=5,
            extended=args.extended
        )
        if success:
            print(f"Successfully processed queries from {args.queries_file} and saved to {args.output}")
        else:
            print(f"Failed to process queries from {args.queries_file}")
    else:
        print("No queries file provided. Use --queries_file to specify a JSON file with queries.")

if __name__ == "__main__":
    main()