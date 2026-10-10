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
    def aggregate(self, weight: list[Any]) :
        """aggregrate logic"""
        pass
    
    @abstractmethod
    def finalize(self) -> list[Any] :
        """return the final updated weight"""
        pass
        
    @abstractmethod
    def reset(self):
        """reset the parameters"""
        pass