from abc import ABC, abstractmethod
from message import Message


class BaseCommunicator(ABC):

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def terminate(self) -> None:
        pass

    @abstractmethod
    def send(self, message: Message, destination: int) -> None:
        pass

    @abstractmethod
    def recv(self, source: int) -> Message:
        pass

    @abstractmethod
    def broadcast(self, message: Message) -> None:
        pass

    @abstractmethod
    def all_gather(self, message: Message) -> list[Message]:
        pass