from abc import ABC, abstractmethod
from base_model import BaseModel


class BaseTrainer(ABC):
    """
    Properties:
        Dataset, Optimizer, Loss Function

    Task:
        Perform local training.

    Input:
        Model

    Return:
        Trained Model
    """

    @abstractmethod
    def train(self, local_epoch: int, model: BaseModel) -> BaseModel:
        pass