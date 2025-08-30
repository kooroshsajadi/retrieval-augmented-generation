import unittest
from src.retrieval.retriever import MilvusRetriever

class TestMilvusRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = MilvusRetriever(collection_name="legal_texts")

    def test_retrieve(self):
        query = "Quali sono i requisiti per la residenza in Italia?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(len(results), 5)
        self.assertIn("chunk_id", results[0])
        self.assertIn("text", results[0])
        self.assertIn("distance", results[0])

if __name__ == "__main__":
    unittest.main()