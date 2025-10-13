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
    
    # Llama 3 8B Instruct: ~8B parameters, auto-regressive transformer (decoder-only) with supervised fine-tuning and RLHF alignment.
    # Designed for high-quality legal dialog, reasoning, and instruction following, supporting multilingual generation and code tasks.
    # Suitable for high-performance servers (GPU with 16GB+ VRAM). Best for legal RAG generation and assistant-style response tasks.
    LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"

    # GPT-2 Large: ~774M parameters, auto-regressive transformer (decoder-only) for general-purpose text generation.
    OPENAI_GPT2_LARGE = "openai-community/gpt2-large"

    # GPT-2 Distil: ~82M parameters, distilled version of GPT-2 for efficient text generation.
    OPENAI_GPT2_DISTIL = "distilbert/distilgpt2"

    # Mistral 7B: Highly efficient causal transformer, open-source, great for limited GPU memory (~7B parameters)
    MISTRAL_7B = "mistralai/mistral-7b"
    
    # Qwen 7B: Strong performance in causal generation, lighter than Llama 3 8B
    QWEN_7B = "Qwen/Qwen-7B"
    
    # Falcon 7B: Popular open-source causal model with 7B params, efficient and effective
    FALCON_7B = "tiiuae/falcon-7b"
    
    # Guanaco 7B: Instruction tuned causal model, good for causal generation and fine-tuning
    GUANACO_7B = "yahma/guanaco-7b"
    
    # Orca Mini 7B: Lightweight causal transformer, known for instruction following
    ORCA_MINI_7B = "kakaobrain/orca_mini_v2_7b"

    # For somewhat larger but still smaller than 8B models (if available GPU memory allows):
    
    # Mistral Mixtral 8x7B: Sparse mixture of experts architecture, improves efficiency
    MIXTRAL_8X7B = "mistralai/mixtral-8x7b"
    
    # Gemma 4B: Smaller causal model with good reasoning for constrained resources
    GEMMA_4B = "gemma/gemma-4b"

    Llama_2_7B = "TheBloke/Llama-2-7B-GPTQ"
