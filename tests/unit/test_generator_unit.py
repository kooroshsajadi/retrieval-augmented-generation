import unittest
from src.generation.generator import LLMGenerator

class TestLLMGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = LLMGenerator(
            model_path="Helsinki-NLP/opus-mt-it-en",
            adapter_path="models/fine_tuned_models/opus-mt-it-en-v1/model",
            tokenizer_path="models/fine_tuned_models/opus-mt-it-en-v1/tokenizer",
            model_type="seq2seq",
            max_length=5000,
            device="xpu"
        )

    def test_generate(self):
        query = "Quali sono i requisiti per la residenza in Italia?"
        contexts = [
            {"chunk_id": 1, "text": "Per la residenza in Italia, è necessario un permesso valido.", "distance": 0.123}
        ]
        response = self.generator.generate(query, contexts, max_new_tokens=50)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Optional: Print for inspection
        print(f"Generated response: {response}")

if __name__ == "__main__":
    unittest.main()