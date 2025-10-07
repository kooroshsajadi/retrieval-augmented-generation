from transformers import AutoTokenizer
from typing import Optional

def create_and_configure_tokenizer(
    model,
    model_name: str,
    tokenizer_path: Optional[str] = None,
    padding_side: str = "left",
    trust_remote_code: bool = False
):
    """
    Load and configure a Hugging Face tokenizer for a model, with logic for missing pad tokens.

    Args:
        model: The model instance, for config and embedding alignment.
        model_name (str): Model identifier (for tokenizer if tokenizer_path is None).
        tokenizer_path (str, optional): Path to alternative tokenizer.
        padding_side (str): Side to pad ('left' recommended for most LLMs).
        trust_remote_code (bool): Whether to trust remote tokenizer code.

    Returns:
        tokenizer: The configured tokenizer object.

    Raises:
        Exception: On failure to load or configure the tokenizer.
    """
    tokenizer_source = tokenizer_path if tokenizer_path is not None else model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            padding_side=padding_side,
            trust_remote_code=trust_remote_code
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
        model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token_id >= model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        return tokenizer
    except Exception:
        raise

def configure_data_type(device: str) -> Optional[str]:
    """
    Determines the appropriate floating-point data type for the given device.

    Parameters
    ----------
    device : str
        The target device identifier. Expected values are:
        - 'cpu': Central Processing Unit
        - 'cuda': NVIDIA GPU (CUDA-enabled)
        - 'xpu': Intel or other accelerators supporting bfloat16

    Returns
    -------
    Optional[str]
        The selected data type as a string:
        - 'float32' for 'cpu'
        - 'float16' for 'cuda'
        - 'bfloat16' for 'xpu'
        - None if the device is unrecognized.

    Examples
    --------
    >>> configure_data_type('cpu')
    'float32'
    >>> configure_data_type('cuda')
    'float16'
    >>> configure_data_type('xpu')
    'bfloat16'
    >>> configure_data_type('unknown')
    None
    """
    device_lower = device.lower()
    if device_lower == 'cpu':
        return 'float32'
    elif device_lower == 'cuda':
        return 'float16'
    elif device_lower == 'xpu':
        return 'bfloat16'
    else:
        return None
