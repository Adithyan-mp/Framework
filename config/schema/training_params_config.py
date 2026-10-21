from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    loss : str 
    epoch : int 
    local_epoch : int
    batch_size : int
    def __post_init__(self):
        if self.local_epoch <= 0:
            raise ValueError(f"Enter valid Local Epoch")
        if self.epoch <=0 :
            raise ValueError(f"Enter valid Epoch")
        if self.batch_size <= 0:
            raise ValueError(f"Enter valid Batch Size")
            

    def create_loss(self):
        try:
            loss_class = getattr(torch.nn, self.loss)
        except AttributeError:
            raise ValueError(
                f"Invalid loss name: {self.loss}"
            ) from None

        if not isinstance(loss_class, type) or not issubclass(
            loss_class, torch.nn.modules.loss._Loss
        ):
            raise ValueError(
                f"'{self.loss}' is not a supported loss class"
            )

        return loss_class()
