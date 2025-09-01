from pathlib import Path
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class ModelLoader:
    """Model loader for inference in RAG pipeline, optimized for seq2seq models."""

    MODEL_TYPE_MAPPING = {
        "seq2seq": AutoModelForSeq2SeqLM
    }

    def __init__(
        self,
        model_name: str = "model/opus-mt-it-en",
        model_type: str = "seq2seq",
        device_map: str = "auto",
        max_length: int = 128,
        logger: logging.Logger = None
    ):
        """
        Initialize ModelLoader for inference.

        Args:
            model_name (str): Path to model or Hugging Face model name.
            model_type (str): Model type ("seq2seq").
            device_map (str): Device placement ("auto", "cpu", "xpu").
            max_length (int): Maximum sequence length for tokenization.
            logger (logging.Logger, optional): Logger instance.
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.max_length = max_length
        self.logger = logger or setup_logger(__name__)

        if self.model_type not in self.MODEL_TYPE_MAPPING:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from {list(self.MODEL_TYPE_MAPPING.keys())}")

        # Set device
        self.use_gpu = torch.cuda.is_available() or torch.xpu.is_available()
        self.use_xpu = torch.xpu.is_available()
        if device_map == "auto":
            self.device = torch.device("xpu" if self.use_xpu else "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_map)
        self.dtype = torch.bfloat16 if self.use_xpu else torch.float16 if self.use_gpu else torch.float32
        self.logger.info(f"Using device: {self.device} with dtype {self.dtype}")

        # Load model
        try:
            model_class = self.MODEL_TYPE_MAPPING[self.model_type]
            self.model = model_class.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=self.dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            ).to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

        # Optimize for Intel ARC
        if self.use_xpu:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                self.logger.info("Applied IPEX optimization for Intel ARC")
            except ImportError:
                self.logger.warning("intel-extension-for-pytorch not installed; skipping IPEX optimization")

        self.model.eval()

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side='left',
                trust_remote_code=False
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
                self.logger.info(f"Set pad_token to {self.tokenizer.pad_token}")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id >= self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.logger.info(f"Resized model embeddings to {len(self.tokenizer)} to accommodate pad_token")
            self.logger.info(f"Set model.config.pad_token_id to {self.model.config.pad_token_id}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
            raise

        self._log_model_profile()

    def _format_count(self, n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n/1_000:.2f}K"
        return str(n)

    def _dtype_num_bytes(self, dtype: torch.dtype) -> int:
        if dtype == torch.float32:
            return 4
        if dtype in (torch.float16, torch.bfloat16):
            return 2
        if dtype in (torch.int8,):
            return 1
        return 4

    def _estimate_param_memory_bytes(self, model: torch.nn.Module) -> int:
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * self._dtype_num_bytes(p.dtype)
        return total_bytes

    def _log_model_profile(self, title: str = "Loaded model profile") -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        approx_param_mem = self._estimate_param_memory_bytes(self.model)
        mem_mb = approx_param_mem / (1024**2)
        mem_gb = approx_param_mem / (1024**3)
        self.logger.info(
            f"\n{title}\n"
            f"- Model name: {self.model_name}\n"
            f"- Model type: {self.model_type}\n"
            f"- Device: {self.device}\n"
            f"- Dtype: {self.dtype}\n"
            f"- Total params: {self._format_count(total_params)} ({total_params:,})\n"
            f"- Approx parameter memory: {mem_mb:.2f}MB ({mem_gb:.3f}GB)"
        )

    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Generate translation or response for seq2seq model."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate: {str(e)}")
            raise