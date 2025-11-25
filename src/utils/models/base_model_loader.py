from abc import ABC, abstractmethod
from typing import Union
import torch

class BaseModelLoaderInterface(ABC):
    @abstractmethod
    def generate(self, text: str, max_new_tokens: int = 50) -> Union[str, torch.Tensor]:
        pass
