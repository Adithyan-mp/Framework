from abc import ABC,abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass