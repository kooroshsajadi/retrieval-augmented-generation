from models.embeddings import SentenceTransformerEmbedding
from models.retriever import VectorStoreRetriever
from models.generator import HuggingFaceGenerator
from models.rag_pipeline import RAGPipeline
from data.vector_store import VectorStore  # Assume data layer provides this

def main():
    # Initialize components
    embedding_model = SentenceTransformerEmbedding()
    vector_store = VectorStore()  # From data layer
    retriever = VectorStoreRetriever(embedding_model, vector_store)
    generator = HuggingFaceGenerator(model_name="gpt2")
    
    # Create RAG pipeline
    rag = RAGPipeline(retriever, generator, top_k=3)
    
    # Test query
    query = "What is machine learning?"
    response = rag.run(query)
    print(f"Query: {query}\nResponse: {response}")

if __name__ == "__main__":
    main()