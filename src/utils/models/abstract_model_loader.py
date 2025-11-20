from abc import ABC, abstractmethod
from typing import Optional, Union
import torch
import logging
from transformers import AutoTokenizer, AutoConfig
from src.utils.logging_utils import setup_logger
from peft import PeftModel

class AbstractModelLoader(ABC):
    """Abstract base class for model loaders in RAG pipeline."""

    model_type: str  # To be defined in subclasses
    model_class: type  # To be defined in subclasses
    expected_configs: tuple  # To be defined in subclasses

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the abstract model loader for inference.

        Args:
            model_name (str): Path to model or Hugging Face model name.
            device_map (str): Device placement ("auto", "cpu", "xpu").
            adapter_path (Optional[str]): Path to LoRA adapters for fine-tuning.
            tokenizer_path (Optional[str]): Path to tokenizer, defaults to model_name.
            max_length (int): Maximum sequence length for tokenization.
            logger (logging.Logger, optional): Logger instance.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.adapter_path = adapter_path
        self.logger = logger or setup_logger("src.utils.models.abstract_model_loader")

        # Detect model config and validate against expected for this subclass
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            if not isinstance(config, self.expected_configs):
                raise ValueError(
                    f"Model {model_name} has config {type(config).__name__}, which does not match expected configs for {self.model_type}: {self.expected_configs}."
                )
        except Exception as e:
            self.logger.error(f"Failed to load config for {model_name}: {str(e)}")
            raise

        # Set device
        self.use_gpu = torch.cuda.is_available() or torch.xpu.is_available()
        self.use_xpu = torch.xpu.is_available()
        if device_map == "auto":
            self.device = torch.device("xpu" if self.use_xpu else "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_map)
        self.dtype = torch.bfloat16 if self.use_xpu else torch.float16 if self.use_gpu else torch.float32
        self.logger.info(f"Using device: {self.device} with dtype {self.dtype}")

        # Load base model
        try:
            base_model = self.model_class.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=self.dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            ).to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

        # Load LoRA adapters if provided
        if self.adapter_path:
            try:
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path).to(self.device)
                self.logger.info(f"Loaded base model {model_name} with adapter from {self.adapter_path}")
            except Exception as e:
                self.logger.error(f"Failed to load adapter model from {self.adapter_path}: {str(e)}")
                raise
        else:
            self.model = base_model
            self.logger.info(f"Loaded base model {model_name} without adapter")

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
        tokenizer_source = tokenizer_path if tokenizer_path is not None else model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                padding_side='left',
                trust_remote_code=False,
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
            f"- Adapter path: {self.adapter_path}\n"
            f"- Device: {self.device}\n"
            f"- Dtype: {self.dtype}\n"
            f"- Total params: {self._format_count(total_params)} ({total_params:,})\n"
            f"- Approx parameter memory: {mem_mb:.2f}MB ({mem_gb:.3f}GB)"
        )

    @abstractmethod
    def generate(self, text: str, max_new_tokens: int = 50) -> Union[str, torch.Tensor]:
        """
        Generate output based on the model type.

        Args:
            text (str): Input text.
            max_new_tokens (int): Maximum new tokens for generation (if applicable).

        Returns:
            Union[str, torch.Tensor]: Generated text or embeddings.
        """
        pass