from enum import Enum

class CrossEncoderModels(Enum):
    # Cross-encoder for reranking, outputs scalar score (not embeddings), suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM)
    MS_MARCO_MINILM_L12_V2 = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Cross-encoder for reranking, outputs scalar score (not embeddings), optimized for Italian, suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM)
    MMARCO_MINILM_L12_H384_V1 = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"