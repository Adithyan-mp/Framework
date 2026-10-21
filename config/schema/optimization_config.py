from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OptimConfig:
    optimizer: str
    lr: float
    args: dict[str, Any] = field(default_factory=dict)

    def create_optim(self, params):
        try:
            optim_class = getattr(torch.optim, self.optimizer)
        except AttributeError:
            raise ValueError(
                f"Invalid optimizer name: {self.optimizer}"
            ) from None
            
        if not isinstance(optim_class,type) or not issubclass(
            optim_class,torch.optim.Optimizer
        ):
            raise ValueError(f"Failed to create optimizer {self.optimizer}")
            
        try:
            return optim_class(
                params,
                lr=self.lr,
                **self.args,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create optimizer "
                f"'{self.optimizer}': {e}"
            ) from e