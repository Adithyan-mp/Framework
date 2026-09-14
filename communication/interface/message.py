from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class Message:
    """
        The message structure to sent using communicator
    """
    message_type: str 
    payload: torch.Tensor | None = None
    source:int | None = None
    destination:int | None = None