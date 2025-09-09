import unittest
from src.retrieval.retriever import MilvusRetriever

class TestMilvusRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = MilvusRetriever(
            collection_name="gotmat_collection",
            embedding_model="intfloat/multilingual-e5-large",
            milvus_host="localhost",
            milvus_port="19530"
        )

    def test_retrieve(self):
        query = "Quali sono i requisiti per la residenza in Italia?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(len(results), 5)

        # Print retrieved chunks for debugging and inspection
        print("\nRetrieved Chunks:")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. chunk_id: {chunk['chunk_id']}")
            print(f"   distance: {chunk['distance']:.4f}")
            snippet = chunk['text'][:200].replace('\n', ' ')  # print first 200 chars without newlines
            print(f"   text snippet: \"{snippet}...\"\n")

        self.assertIn("chunk_id", results[0])
        self.assertIn("text", results[0])
        self.assertIn("distance", results[0])

if __name__ == "__main__":
    unittest.main()