from typing import List, Tuple
from .retriever import Retriever
from .generator import Generator

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> str:
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, self.top_k)
        contexts = [doc for doc, _ in retrieved_docs]  # Extract document texts
        
        # Step 2: Generate response using contexts
        response = self.generator.generate(query, contexts)
        return response