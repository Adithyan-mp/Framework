from abc import ABC, abstractmethod
from typing import Any


class BaseAggregator(ABC):
    """
        Properties:
            None
        Task:
            Take models weight and preform aggregation
        Return:
            aggregated weight
    """

    @abstractmethod
    def aggregate(self, weights: list[Any]) -> Any:
        pass