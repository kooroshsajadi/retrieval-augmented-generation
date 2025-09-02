import unittest
import os
from src.scripts.main import RAGOrchestrator

class TestRAGOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config_path = "configs/rag.yaml"
        self.orchestrator = RAGOrchestrator(self.config_path)

    def test_process_query(self):
        query = "Quali sono i requisiti per la residenza in Italia?"
        result = self.orchestrator.process_query(query, top_k=5, max_new_tokens=50)
        self.assertEqual(result["query"], query)
        self.assertIsInstance(result["contexts"], list)
        self.assertEqual(len(result["contexts"]), 5)
        self.assertIsInstance(result["response"], str)
        self.assertGreater(len(result["response"]), 0)

    def test_process_batch(self):
        queries = [
            "Quali sono i requisiti per la residenza in Italia?",
            "Come ottenere un visto di lavoro?"
        ]
        results = self.orchestrator.process_batch(queries, top_k=5, max_new_tokens=50)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn(result["query"], queries)
            self.assertIsInstance(result["contexts"], list)
            self.assertIsInstance(result["response"], str)

if __name__ == "__main__":
    unittest.main()