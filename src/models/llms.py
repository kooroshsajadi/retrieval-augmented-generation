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
    
    # Italian-Legal-BERT: ~110M parameters, encoder-only transformer fine-tuned on Italian legal texts.
    # Suitable for text classification, named entity recognition, or generating embeddings for legal documents.
    # Compatible with medium to high-performance machines (CPU or GPU with 4GB+ VRAM).
    ITALIAN_LEGAL_BERT = "dlicari/Italian-Legal-BERT"
    
    # Italian-Legal-BERT-SC: ~110M parameters, encoder-only transformer fine-tuned for semantic classification on Italian legal texts.
    # Ideal for understanding legal context or generating embeddings for legal chunks in the RAG pipeline.
    # Compatible with medium to high-performance machines (CPU or GPU with 4GB+ VRAM).
    ITALIAN_LEGAL_BERT_SC = "dlicari/Italian-Legal-BERT-SC"
    
    # Distil-Italian-Legal-BERT: ~67M parameters, encoder-only transformer (distilled BERT) fine-tuned on Italian legal texts.
    # Lightweight model for embedding generation or classification tasks with lower resource demands.
    # Suitable for low to medium-performance machines (CPU or GPU with 2GB+ VRAM).
    DISTIL_ITA_LEGAL_BERT = "dlicari/distil-ita-legal-bert"