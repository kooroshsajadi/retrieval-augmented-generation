from enum import Enum   

class LargeLanguageModels(Enum):
    # mBART-large-50: ~610M parameters, encoder-decoder transformer for multilingual sequence-to-sequence tasks.
    # Primarily used for translation (e.g., Italian to English) in response generation for legal texts.
    # Suitable for high-performance servers (GPU with 12GB+ VRAM).
    MBART_LARGE_50 = "facebook/mbart-large-50"
    
    # OPUS-MT-IT-EN: ~78M parameters, encoder-decoder transformer optimized for Italian-to-English translation.
    # Lightweight and efficient for translating legal queries or responses in the RAG pipeline.
    # Suitable for medium-performance machines (CPU or GPU with 4GB+ VRAM).
    OPUS_MT_IT_EN = "Helsinki-NLP/opus-mt-it-en"
    
    # Distil-Italian-Legal-BERT: ~67M parameters, encoder-only transformer (distilled BERT) fine-tuned on Italian legal texts.
    # Lightweight model for embedding generation or classification tasks with lower resource demands.
    # Suitable for low to medium-performance machines (CPU or GPU with 2GB+ VRAM).
    DISTIL_ITA_LEGAL_BERT = "dlicari/distil-ita-legal-bert"
    
    # Llama 3 8B Instruct: ~8B parameters, auto-regressive transformer (decoder-only) with supervised fine-tuning and RLHF alignment.
    # Designed for high-quality legal dialog, reasoning, and instruction following, supporting multilingual generation and code tasks.
    # Suitable for high-performance servers (GPU with 16GB+ VRAM). Best for legal RAG generation and assistant-style response tasks.
    LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
