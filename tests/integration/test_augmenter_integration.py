import unittest
import logging
from src.augmentation.augmenter import Augmenter

class TestAugmenterIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("augmenter_test")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.augmenter = Augmenter(max_contexts=2, max_context_length=500, logger=self.logger)
        self.query = "Quali sono i requisiti per la residenza in Italia?"
        self.contexts = [
            {"chunk_id": "116876_chunk_0", "text": "Per la residenza in Italia, è necessario un passaporto valido...", "distance": 0.12, "subject": "courthouse"},
            {"chunk_id": "116876_chunk_1", "text": "I requisiti includono un'assicurazione sanitaria valida...", "distance": 0.15, "subject": "courthouse"}
        ]

    def test_augment(self):
        prompt = self.augmenter.augment(self.query, self.contexts)
        self.assertIn("Query: Quali sono i requisiti", prompt)
        self.assertIn("Context:", prompt)
        self.assertIn("(courthouse) Per la residenza in Italia", prompt)
        self.assertIn("(courthouse) I requisiti includono", prompt)
        self.logger.info("Augmentation test passed")

if __name__ == "__main__":
    unittest.main()