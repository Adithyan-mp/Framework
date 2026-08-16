from abc import ABC, abstractmethod


class BaseTopology(ABC):
    """
        define how the entire system is connected together
    """

    @abstractmethod
    def get_neighbours(self, node: int) -> list[int]:
        pass

    @abstractmethod
    def is_connected(self, node1: int, node2: int) -> bool:
        pass