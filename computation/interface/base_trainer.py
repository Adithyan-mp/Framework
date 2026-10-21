from abc import ABC, abstractmethod
from computation.interface.base_model import BaseModel


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
    def train(self, dataloader,local_epoch: int, model: BaseModel) -> None:
        pass