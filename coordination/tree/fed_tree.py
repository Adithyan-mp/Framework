
from coordination.interface.base_scheduler import BaseScheduler

from computation.trainer.local_trainer import LocalTrainer

from communication.communicator.torch_communicator import (
    TorchCommunicator,
)
from communication.interface.base_communicator import BaseCommunicator
from communication.interface.base_topology import BaseTopology
from communication.interface.message import Message
from communication.topology.tree_topology import TreeTopology

from computation.interface.base_trainer import BaseTrainer
from computation.interface.base_model import BaseModel

import torch


class FedTree(BaseScheduler):

    def __init__(
        self,
        model: BaseModel,
        config: str,
        trainer: BaseTrainer = None,
        communicator: BaseCommunicator = None,
        topology: BaseTopology = None,
    ):
        super().__init__(model, config)

        self.train_params = self.config.load_train_params()

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            optim_config = self.config.load_torch_optim_config()

            optimizer = optim_config.create_optim(
                self.model.get_parameters()
            )

            self.trainer = LocalTrainer(
                self.train_params.create_loss(),
                optimizer,
            )

        # Communicator
        self.communicator = (
            communicator
            if communicator is not None
            else TorchCommunicator(self.config)
        )

        # Topology
        self.topology = (
            topology
            if topology is not None
            else TreeTopology(
                self.config.load_tree_topology().data
            )
        )

    # --------------------------------------------------
    # Model weight helpers
    # --------------------------------------------------

    def _get_weights(self) -> list[torch.Tensor]:
        """Return independent copies of the model state tensors."""

        return [
            tensor.detach().clone()
            for tensor in self.model.get_weights().values()
        ]

    def _set_weights(self, weights: list[torch.Tensor]) -> None:
        """Copy received tensors into the model."""

        model_tensors = list(self.model.get_weights().values())

        if len(model_tensors) != len(weights):
            raise ValueError(
                "Mismatch in the number of model tensors"
            )

        with torch.no_grad():
            for model_tensor, received_tensor in zip(
                model_tensors, weights
            ):
                if model_tensor.shape != received_tensor.shape:
                    raise ValueError(
                        "Model tensor shape mismatch"
                    )

                if model_tensor.dtype != received_tensor.dtype:
                    raise ValueError(
                        "Model tensor dtype mismatch"
                    )

                model_tensor.copy_(received_tensor)

    # --------------------------------------------------
    # Sample count helpers
    # --------------------------------------------------

    @staticmethod
    def _get_sample_count(dataloader) -> int:
        """Get the number of local training examples."""

        if not hasattr(dataloader, "dataset"):
            raise ValueError(
                "Dataloader must expose its dataset"
            )

        count = len(dataloader.dataset)

        if count <= 0:
            raise ValueError(
                "Local training dataset must not be empty"
            )

        return count

    @staticmethod
    def _make_count_tensor(
        count: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(
            count,
            dtype=torch.int64,
            device=reference.device,
        )

    # --------------------------------------------------
    # Message helpers
    # --------------------------------------------------

    def _send_subtree(
        self,
        weights: list[torch.Tensor],
        sample_count: int,
        destination: int,
    ) -> None:
        """Send aggregated weights and represented sample count."""

        count_tensor = self._make_count_tensor(
            sample_count,
            weights[0],
        )

        payload = [
            tensor.detach().contiguous()
            for tensor in weights
        ]
        payload.append(count_tensor)

        message = Message(
            message_type="MODEL_WEIGHTS",
            payload=payload,
            source=self.communicator.get_rank(),
            destination=destination,
        )

        self.communicator.send(message)

    def _receive_subtree(
        self,
        source: int,
    ) -> tuple[list[torch.Tensor], int]:
        """Receive aggregated weights and sample count."""

        weights = self._get_weights()

        if not weights:
            raise ValueError("Model contains no state tensors")

        buffers = [
            torch.empty_like(tensor)
            for tensor in weights
        ]

        count_buffer = self._make_count_tensor(
            0,
            buffers[0],
        )

        buffers.append(count_buffer)

        message = Message(
            message_type="MODEL_WEIGHTS",
            payload=buffers,
            source=source,
        )

        self.communicator.recv(message)

        received_weights = buffers[:-1]
        sample_count = int(buffers[-1].item())

        if sample_count <= 0:
            raise ValueError(
                f"Invalid sample count received from rank {source}"
            )

        return received_weights, sample_count

    # --------------------------------------------------
    # Hierarchical weighted aggregation
    # --------------------------------------------------

    @staticmethod
    def _aggregate_subtrees(
        local_weights: list[torch.Tensor],
        local_count: int,
        child_results: list[
            tuple[list[torch.Tensor], int]
        ],
    ) -> tuple[list[torch.Tensor], int]:
        """
        Combine local model weights with child subtree averages.

        Floating-point and complex tensors are sample-weighted.
        Non-floating tensors retain the local node's values.
        """

        total_count = local_count

        for _, child_count in child_results:
            total_count += child_count

        aggregated = []

        for tensor_idx, local_tensor in enumerate(local_weights):

            # Integer buffers cannot be averaged meaningfully.
            if not (
                local_tensor.is_floating_point()
                or local_tensor.is_complex()
            ):
                aggregated.append(local_tensor.clone())
                continue

            result = local_tensor * local_count

            for child_weights, child_count in child_results:
                child_tensor = child_weights[tensor_idx]

                if child_tensor.shape != local_tensor.shape:
                    raise ValueError(
                        "Shape mismatch in child model tensors"
                    )

                result = result + child_tensor * child_count

            result = result / total_count
            aggregated.append(result)

        return aggregated, total_count

    # --------------------------------------------------
    # Broadcast global model
    # --------------------------------------------------

    def _broadcast_global_weights(self, root: int) -> None:
        """All ranks must call this method in the same round."""

        weights = self._get_weights()

        message = Message(
            message_type="GLOBAL_MODEL",
            payload=weights,
            source=root,
        )

        self.communicator.broadcast(message)

        # Broadcast modifies the message payload buffers.
        self._set_weights(weights)

    # --------------------------------------------------
    # Federated training
    # --------------------------------------------------

    def run(self, dataloader=None):

        if dataloader is None:
            raise ValueError("Need a valid DataLoader")

        self.communicator.init()

        rank = self.communicator.get_rank()
        root = self.topology.root
        children = self.topology.get_children(rank)

        local_count = self._get_sample_count(dataloader)

        try:
            # Distribute the initial global model.
            self._broadcast_global_weights(root)

            for round_idx in range(self.train_params.epoch):

                print(
                    f"[Rank {rank}] Starting round "
                    f"{round_idx + 1}"
                )

                # 1. Train on local data.
                self.trainer.train(
                    dataloader=dataloader,
                    local_epoch=self.train_params.local_epoch,
                    model=self.model,
                )

                local_weights = self._get_weights()

                # 2. Receive aggregated results from children.
                child_results = []

                for child in children:
                    child_weights, child_count = (
                        self._receive_subtree(child)
                    )

                    child_results.append(
                        (child_weights, child_count)
                    )

                # 3. Aggregate local and child model weights.
                aggregated_weights, subtree_count = (
                    self._aggregate_subtrees(
                        local_weights=local_weights,
                        local_count=local_count,
                        child_results=child_results,
                    )
                )

                self._set_weights(aggregated_weights)

                print(
                    f"[Rank {rank}] Aggregated "
                    f"{subtree_count} training examples"
                )

                # 4. Send the subtree result to the parent.
                if rank != root:
                    parent = self.topology.get_parent(rank)

                    self._send_subtree(
                        weights=aggregated_weights,
                        sample_count=subtree_count,
                        destination=parent,
                    )

                # 5. Root now has the global aggregate.
                # Every rank participates in the next broadcast.
                self._broadcast_global_weights(root)

                print(
                    f"[Rank {rank}] Finished round "
                    f"{round_idx + 1}"
                )

        finally:
            self.communicator.terminate()
