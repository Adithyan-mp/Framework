from abc import ABC, abstractmethod
from communication.interface.message import Message
import torch


class BaseCommunicator(ABC):

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def terminate(self) -> None:
        pass

    @abstractmethod
    def send(self, message: Message) -> None:
        pass

    @abstractmethod
    def recv(self, message: Message) -> Message:
        pass

    @abstractmethod
    def broadcast(self, message: Message) -> None:
        pass

    @abstractmethod
    def all_gather(self, message: Message) -> list[torch.Tensor]:
        pass