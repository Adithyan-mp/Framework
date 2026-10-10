from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OptimConfig:
    name: str
    lr: float
    args: dict[str, Any] = field(default_factory=dict)

    def create_optim(self, params):
        try:
            optim_class = getattr(torch.optim, self.name)
        except AttributeError:
            raise ValueError(
                f"Invalid optimizer name: {self.name}"
            ) from None

        try:
            return optim_class(
                params,
                lr=self.lr,
                **self.args
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create optimizer '{self.name}': {e}"
            ) from e