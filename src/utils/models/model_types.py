from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from enum import Enum
from src.models.causal_model_loader import CausalModelLoader
from src.models.encoder_only_model_loader import EncoderOnlyModelLoader
from src.models.seq2seq_model_loader import Seq2SeqModelLoader
from src.models.ollama_model_loader import OllamaModelLoader


MODEL_LOADER_MAPPING = {
    "seq2seq": Seq2SeqModelLoader,
    "encoder-only": EncoderOnlyModelLoader,
    "causal": CausalModelLoader,
    "ollama": OllamaModelLoader
}

class ModelTypes(Enum):
    SEQ2SEQ = "seq2seq"
    ENCODER_ONLY = "encoder-only"
    CASUAL = "causal"  # For models like GPT, LLaMA, etc.
