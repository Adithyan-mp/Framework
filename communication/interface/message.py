from dataclasses import dataclass
from typing import Any

@dataclass
class Message:
    """
        The message structure to sent using communicator
    """
    message_type: str
    payload: Any