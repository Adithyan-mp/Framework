
from abc import ABC, abstractmethod
from communication.interface.message import Message
import torch


TensorPayload = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]


class BaseCommunicator(ABC):

    @abstractmethod
    def init(self) -> None:
        """Initialize the communication backend."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Release communication resources."""
        pass

    @abstractmethod
    def send(self, message: Message) -> None:
        """Send a tensor or a collection of tensors."""
        pass

    @abstractmethod
    def recv(self, message: Message) -> Message:
        """Receive tensors into the supplied payload buffers."""
        pass

    @abstractmethod
    def broadcast(self, message: Message) -> None:
        """Broadcast a tensor or collection of tensors."""
        pass

    @abstractmethod
    def all_gather(
        self, message: Message
    ) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        """Gather tensors from all ranks.

        Single tensor payload:
            list[torch.Tensor]

        Multiple tensor payload:
            list[list[torch.Tensor]], indexed by rank.
        """
        pass
