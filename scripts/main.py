import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logging_utils import setup_logger
from scripts.validate_data import DataValidator
from scripts.ingest_data import DataIngestor
from scripts.sentence_transformer import SentenceTransformerEmbedder
from src.data.vector_store import VectorStore
from src.retrieval.retriever import MilvusRetriever
from src.generation.generator import LLMGenerator

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
        self.text_cleaner = TextCleaner(logger=self.logger)
        self.text_chunker = TextChunker(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            logger=self.logger
        )
        self.embedder = SentenceTransformerEmbedder(
            model_name=self.config.get("embedding_model", "intfloat/multilingual-e5-large"),
            output_dir=self.config["data"]["embeddings"],
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            logger=self.logger
        )
        self.vector_store = VectorStore(
            collection_name=self.config.get("collection_name", "legal_texts"),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
            logger=self.logger
        )
        self.retriever = MilvusRetriever(
            collection_name=self.config.get("collection_name", "legal_texts"),
            embedding_model=self.config.get("embedding_model", "intfloat/multilingual-e5-large"),
            milvus_host=self.config.get("milvus_host", "localhost"),
            milvus_port=self.config.get("milvus_port", "19530"),
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
                self.logger.error(f"File validation failed: {validation_result['error']}")
                return False

            # Extract text
            ingest_result = self.data_ingestor.extract_text(file_path)
            if not ingest_result["is_valid"]:
                self.logger.error(f"Text extraction failed: {ingest_result['error']}")
                return False

            # Clean text
            cleaned_text = self.text_cleaner.clean_text(ingest_result["text"])
            cleaned_text_path = Path(self.config["data"]["cleaned_texts"]) / f"{Path(file_path).stem}_cleaned.txt"
            with open(cleaned_text_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            self.logger.info(f"Saved cleaned text to {cleaned_text_path}")

            # Chunk text
            chunks = self.text_chunker.chunk_text(cleaned_text)
            self.logger.info(f"Generated {len(chunks)} chunks")

            # Generate embeddings and store in Milvus
            embed_result = self.embedder.process_file(file_path, cleaned_text)
            if not embed_result["is_valid"]:
                self.logger.error(f"Embedding generation failed: {embed_result['error']}")
                return False
            chunks = [c["text"] for c in embed_result["chunk_embeddings"]]
            embeddings = [np.load(self.config["data"]["embeddings"] / c["embedding_file"]) for c in embed_result["chunk_embeddings"]]
            self.vector_store.store_vectors(chunks, embeddings)
            self.logger.info(f"Stored {len(chunks)} embeddings in Milvus")
            return True
        except Exception as e:
            self.logger.error(f"File processing failed for {file_path}: {str(e)}")
            return False

    def process_query(self, query: str, top_k: int = 5) -> str:
        """
        Process a user query and generate a response.

        Args:
            query (str): User query in Italian.
            top_k (int): Number of chunks to retrieve.

        Returns:
            str: Generated response in Italian.
        """
        try:
            # Generate query embedding
            query_result = self.embedder.process_query(query)
            if not query_result["is_valid"]:
                self.logger.error(f"Query embedding failed: {query_result['error']}")
                return f"Error: {query_result['error']}"

            # Retrieve relevant chunks
            contexts = self.retriever.retrieve(query, top_k)
            self.logger.info(f"Retrieved {len(contexts)} contexts for query: {query[:50]}...")

            # Generate response
            response = self.generator.generate(query, contexts, max_new_tokens=self.config.get("max_new_tokens", 50))
            self.logger.info(f"Generated response: {response[:100]}...")

            # Save response
            output_path = Path(self.config["data"]["destination"]) / f"response_{hash(query)}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"query": query, "response": response, "contexts": contexts}, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved response to {output_path}")
            return response
        except Exception as e:
            self.logger.error(f"Query processing failed for '{query}': {str(e)}")
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Orchestrator")
    parser.add_argument("--query", type=str, required=True, help="User query in Italian")
    parser.add_argument("--file", type=str, help="Path to optional input file (PDF, text, or image)")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to configuration file")
    args = parser.parse_args()

    orchestrator = RAGOrchestrator(config_path=args.config)
    
    # Process file if provided
    if args.file:
        success = orchestrator.process_file(args.file)
        if not success:
            print(f"Failed to process file: {args.file}")
            return

    # Process query
    response = orchestrator.process_query(args.query)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()