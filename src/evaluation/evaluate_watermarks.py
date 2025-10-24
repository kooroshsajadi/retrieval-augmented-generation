from pathlib import Path
import yaml
import json
import torch
from src.utils.logging_utils import setup_logger
from src.generation.watermark_processor import WatermarkDetector
from src.utils.models.model_utils import create_and_configure_tokenizer
from src.utils.models.model_types import MODEL_LOADER_MAPPING, ModelTypes
import numpy as np

class WatermarkEvaluator:
    def __init__(self, model_path: str
                 , model_type: str = ModelTypes.CASUAL.value
                 , adapter_path: str | None = None
                 , tokenizer_path: str | None = None
                 , max_length: int = 2048
                 , device: str = "auto", logger=None):
        self.logger = logger or setup_logger("src.evaluation.evaluate_watermarks")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))
        self.model_type = model_type
        self.max_length = max_length
        self.logger.info(f"Initialized WatermarkEvaluator with device: {self.device}, model_type: {model_type}")

        # Initialize the model loader and tokenizer
        try:
            self.model_loader = MODEL_LOADER_MAPPING[self.model_type](
                model_name=model_path,
                device_map=self.device,
                adapter_path=adapter_path,
                tokenizer_path=tokenizer_path,
                max_length=self.max_length
            )
            self.model = self.model_loader.model
            tokenizer_source = tokenizer_path if tokenizer_path is not None else model_path
            self.tokenizer = create_and_configure_tokenizer(
                model=self.model,
                model_name=model_path,
                tokenizer_path=tokenizer_source
            )
            self.logger.info(f"Loaded tokenizer for model: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer from {model_path}: {str(e)}")
            raise

        # Initialize watermark detector
        try:
            self.watermark_detector = WatermarkDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=0.25,  # should match original setting
                seeding_scheme="selfhash",  # should match original setting
                device=self.device,  # use the resolved device
                tokenizer=self.tokenizer,
                z_threshold=4.0,
                normalizers=[],
                ignore_repeated_ngrams=True
            )
            self.logger.info("Initialized WatermarkDetector")
        except Exception as e:
            self.logger.error(f"Failed to initialize WatermarkDetector: {str(e)}")
            raise

    def load_rag_results(self, rag_results_path: str) -> list:
        """
        Load RAG pipeline results from a JSON file.

        Args:
            rag_results_path (str): Path to the RAG results JSON file.

        Returns:
            list: List of RAG result dictionaries.

        Raises:
            FileNotFoundError: If the RAG results file is not found.
            Exception: For other errors during file loading.
        """
        try:
            with open(rag_results_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            self.logger.info(f"Loaded RAG results from {rag_results_path}")
            return dataset
        except FileNotFoundError:
            self.logger.error(f"RAG results file not found: {rag_results_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load RAG results from {rag_results_path}: {str(e)}")
            raise

    def evaluate(self, dataset: list) -> list:
        """
        Evaluate watermark presence in RAG pipeline results.

        Args:
            dataset (list): List of RAG result dictionaries containing 'query' and 'answer'.

        Returns:
            list: List of evaluation results with watermark scores.
        """
        results = []
        for data in dataset:
            self.logger.info(f"Processing query: {data['query'][:50]}...")
            watermark_scores = self.watermark_detector.detect(data['answer'])
            print(f"Query: {data['query'][:50]}...\nWatermark Scores: {watermark_scores}\n")
            
            # Make a copy and convert non-serializable types
            scores_copy = watermark_scores.copy()
            if 'z_score_at_T' in scores_copy and isinstance(scores_copy['z_score_at_T'], torch.Tensor):
                scores_copy['z_score_at_T'] = scores_copy['z_score_at_T'].tolist()
            for key, value in scores_copy.items():
                if isinstance(value, np.float64):
                    scores_copy[key] = float(value)
            
            results.append({
                'query': data['query'],
                'answer': data['answer'],
                'watermark_scores': scores_copy
            })
        self.logger.info(f"Evaluated {len(results)} items")
        return results

    def save_evaluation_results(self, results: list, output_path: str) -> None:
        """
        Save watermark evaluation results to a JSON file.

        Args:
            results (list): List of evaluation results.
            output_path (str): Path to save the evaluation results JSON file.

        Raises:
            Exception: For errors during file saving.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved watermark evaluation results to {output_path}")
            print(f"Saved watermark evaluation results to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save watermark evaluation results to {output_path}: {str(e)}")
            raise

if __name__ == "__main__":
    with open(Path("configs/rag.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize evaluator with model path and configuration
    evaluator = WatermarkEvaluator(
        model_path=config['model']['model_path'],
        model_type=config['model'].get('model_type', ModelTypes.CASUAL.value),
        adapter_path=config['model'].get('adapter_path', None),
        tokenizer_path=config['model'].get('tokenizer_path', None),
        max_length=config.get('max_input_tokenization_length', 2048),
        device="auto"
    )

    # Load RAG results
    rag_results_path = "data/results/responses_(leggi_area3)(reranking_bm25_deduplication)(repetition_penalty_sampling_temperature_topp)_falcon7binstruct_finetuned_extended.json"
    dataset = evaluator.load_rag_results(rag_results_path)

    # Evaluate and save results
    results = evaluator.evaluate(dataset)
    output_path = "data/evaluation/watermarking_falcon7binstructgenerator_distilledgpt2evaluator.json"
    evaluator.save_evaluation_results(results, output_path)