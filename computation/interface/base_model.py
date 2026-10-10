from abc import ABC, abstractmethod
import torch


class BaseModel(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        pass

    def get_parameters(self):
        return self.parameters()

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)