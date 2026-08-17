from abc import ABC, abstractmethod

from computation.interface.base_trainer import BaseTrainer
from computation.interface.base_aggregator import BaseAggregator
from communication.interface.base_communicator import BaseCommunicator


class BaseScheduler(ABC):
    """
    Properties:
        Trainer
        Aggregator
        Communicator

    Task:
        Maintain the overall FL workflow.
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        aggregator: BaseAggregator,
        communicator: BaseCommunicator
    ):
        self.trainer = trainer
        self.aggregator = aggregator
        self.communicator = communicator

    @abstractmethod
    def run(self) -> None:
        pass