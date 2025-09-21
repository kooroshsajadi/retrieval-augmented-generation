from enum import Enum

class EncoderModels(Enum):
    # Embedding size: 1024, max tokens: 512, suitable for high-performance machines (GPU with 8GB+ VRAM).
    # Ideal for generating embeddings for multilingual legal texts in RAG pipeline.
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    
    # Embedding size: 1024, max tokens: 512, suitable for high-performance machines (GPU with 8GB+ VRAM).
    # Instruction-tuned for enhanced semantic embeddings, ideal for legal text queries and documents.
    MULTILINGUAL_E5_LARGE_INSTRUCT = "intfloat/multilingual-e5-large-instruct"
    
    # Embedding size: 768, max tokens: 512, suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM).
    # Fine-tuned on Italian legal texts, ideal for legal-specific embeddings or classification tasks.
    ITALIAN_LEGAL_BERT_SC = "dlicari/Italian-Legal-BERT-SC"

    # Embedding size: 768, max tokens: 512; fine-tuned on large Italian legal corpus (civil cases, judgments, legal codes) using bert-base-italian-xxl-cased as a foundation.
    # ITALIAN_LEGAL_BERT: Delivers superior performance over general BERT for Italian legal research tasks, including textual entailment, argument mining, and document classification.
    # Ideal for embedding generation and supervised learning within the Italian legal domain.
    ITALIAN_LEGAL_BERT = "dlicari/Italian-Legal-BERT"
    
    # Embedding size: 768 (after pooling), max tokens: 512, suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM).
    # BERT-based model for Italian text, suitable for generating embeddings or classification.
    BERT_BASE_ITALIAN_CASED = "dbmdz/bert-base-italian-cased"
    
    # Embedding size: 768, max tokens: 512, suitable for medium to high-performance machines (CPU or GPU with 4GB+ VRAM).
    # Distilled Italian Legal BERT, lightweight and efficient for embedding and reranking tasks in the legal domain.
    DISTIL_ITA_LEGAL_BERT = "dlicari/distil-ita-legal-bert"
    
    # Embedding size: 768, max tokens: 128, suitable for medium-performance machines (CPU or GPU with 4GB+ VRAM).
    # Optimized for paraphrase detection and multilingual embeddings, ideal for shorter legal texts.
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2 = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Embedding size: 512, max tokens: 512, suitable for low-performance machines (CPU or GPU with 2GB+ VRAM).
    # Distilled model for efficient multilingual embeddings, suitable for resource-constrained environments.
    DISTILUSE_BASE_MULTILINGUAL_CASED_V2 = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    
    # Embedding size: 4096, max tokens: 32768, suitable for very high-performance machines (GPU with 24GB+ VRAM or multi-GPU).
    # Large model for high-quality embeddings, ideal for complex legal texts with long contexts.
    E5_MISTRAL_7B_INSTRUCT = "intfloat/e5-mistral-7b-instruct"
    
    # Embedding size: 1024 (after pooling), max tokens: 512, suitable for high-performance machines (GPU with 12GB+ VRAM).
    # Large multilingual model for embeddings, suitable for diverse legal text applications.
    XLM_ROBERTA_LARGE = "xlm-roberta-large"
