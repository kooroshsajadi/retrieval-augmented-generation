import logging
from typing import List, Dict, Any, Optional
from gliner import GLiNER
from src.utils.logging_utils import setup_logger  # Assuming this exists in your pipeline for consistency

class Anonymizer:
    """Handles anonymization of text using a NER model to hide individual names."""

    def __init__(
        self,
        model_name: str = "DeepMount00/universal_ner_ita",
        labels: List[str] = ["persona", "nome"],
        replacement: str = "[PERSON]",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Anonymizer.

        Args:
            model_name (str): Hugging Face NER model name (zero-shot GLiNER-based).
            labels (List[str]): Entity labels to detect (e.g., for names).
            replacement (str): Placeholder to replace detected entities.
            logger (Optional[logging.Logger]): Logger instance.
        """
        self.labels = labels
        self.replacement = replacement
        self.logger = logger or setup_logger("src.utils.anonymizer")
        try:
            self.model = GLiNER.from_pretrained(model_name)
            self.logger.info(f"Loaded NER model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load NER model {model_name}: {str(e)}")
            raise

    def anonymize(self, text: str) -> str:
        """
        Anonymize text by replacing detected entities with a placeholder.

        Args:
            text (str): Input text to anonymize.

        Returns:
            str: Anonymized text.
        """
        try:
            entities = self.model.predict_entities(text, self.labels)
            anonymized_text = text
            # Sort entities by start index descending to avoid index shifts during replacement
            for entity in sorted(entities, key=lambda e: e['start'], reverse=True):
                anonymized_text = (
                    anonymized_text[:entity['start']] + self.replacement + anonymized_text[entity['end']:]
                )
            self.logger.debug(f"Anonymized text (original length: {len(text)}, entities found: {len(entities)})")
            return anonymized_text
        except Exception as e:
            self.logger.error(f"Anonymization failed: {str(e)}")
            return text  # Return original on error

def main():
    # Sample test
    anonymizer = Anonymizer()
    sample_text = "Mario Rossi vive a Roma."
    anonymized = anonymizer.anonymize(sample_text)
    print(f"Original: {sample_text}")
    print(f"Anonymized: {anonymized}")

if __name__ == "__main__":
    main()