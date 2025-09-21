from transformers import AutoModel, AutoModelForSeq2SeqLM
from enum import Enum

MODEL_TYPE_MAPPING = {
    "seq2seq": AutoModelForSeq2SeqLM,
    "encoder-only": AutoModel
}

class ModelTypes(Enum):
    SEQ2SEQ = "seq2seq"
    ENCODER_ONLY = "encoder-only"
    CASUAL = "causal"  # For models like GPT, LLaMA, etc.
