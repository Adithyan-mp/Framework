from abc import ABC, abstractmethod

from computation.interface.base_model import BaseModel
from config.loader.load_config import LoadConfig


class BaseScheduler(ABC):
    """
    Base class for federated learning schedulers.

    Responsibilities:
        - Maintain the model and configuration.
        - Define the interface for executing the FL workflow.
    """

    def __init__(
        self,
        model: BaseModel,
        config: str,
    ):
        self.model = model
        self.config = LoadConfig(fname=config)

    @abstractmethod
    def run(self) -> None:
        """Execute the federated learning workflow."""
        pass