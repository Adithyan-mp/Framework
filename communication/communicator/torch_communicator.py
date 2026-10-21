
from communication.interface.base_communicator import BaseCommunicator
from config.loader.load_config import LoadConfig
from communication.interface.message import Message

import os
import torch
import torch.distributed as dist


class TorchCommunicator(BaseCommunicator):

    def __init__(self, config_loader: LoadConfig):
        self._config_data = (
            config_loader.load_torch_communicator_config()
        )
        self._configure_environment()

    def get_rank(self):
        return self._config_data.rank

    def get_world_size(self):
        return self._config_data.world_size

    def _configure_environment(self):
        if self._config_data.nccl_socket_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = (
                self._config_data.nccl_socket_ifname
            )

        if self._config_data.gloo_socket_ifname:
            os.environ["GLOO_SOCKET_IFNAME"] = (
                self._config_data.gloo_socket_ifname
            )

        os.environ["MASTER_ADDR"] = self._config_data.master_addr
        os.environ["MASTER_PORT"] = str(
            self._config_data.master_port
        )

    def init(self):
        dist.init_process_group(
            backend=self._config_data.backend,
            rank=self._config_data.rank,
            world_size=self._config_data.world_size,
            timeout=self._config_data.timeout,
        )

    def terminate(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _as_tensor_list(payload):
        """Normalize a tensor or list of tensors."""
        if isinstance(payload, torch.Tensor):
            return [payload]

        if isinstance(payload, (list, tuple)):
            if not all(
                isinstance(tensor, torch.Tensor)
                for tensor in payload
            ):
                raise TypeError(
                    "Payload must contain only tensors"
                )
            return list(payload)

        raise TypeError(
            "Payload must be a tensor or a list of tensors"
        )

    @staticmethod
    def _validate_payload(message):
        if message.payload is None:
            raise ValueError("Message payload is required")

    def send(self, message: Message) -> None:
        if message.destination is None:
            raise ValueError("Message destination is required")

        self._validate_payload(message)
        tensors = self._as_tensor_list(message.payload)

        for tensor in tensors:
            dist.send(
                tensor=tensor.contiguous(),
                dst=message.destination,
            )

    def recv(self, message: Message) -> Message:
        if message.source is None:
            raise ValueError("Message source is required")

        self._validate_payload(message)
        tensors = self._as_tensor_list(message.payload)

        with torch.no_grad():
            for tensor in tensors:
                buffer = torch.empty_like(
                    tensor,
                    memory_format=torch.contiguous_format,
                )

                dist.recv(
                    tensor=buffer,
                    src=message.source,
                )

                tensor.copy_(buffer)

        return message

    def broadcast(self, message: Message) -> None:
        if message.source is None:
            raise ValueError("Message source is required")

        self._validate_payload(message)
        tensors = self._as_tensor_list(message.payload)

        for tensor in tensors:
            dist.broadcast(
                tensor=tensor,
                src=message.source,
            )

    def all_gather(self, message: Message):
        self._validate_payload(message)
        tensors = self._as_tensor_list(message.payload)

        gathered_by_rank = [
            [] for _ in range(self._config_data.world_size)
        ]

        for tensor in tensors:
            gathered = [
                torch.empty_like(tensor)
                for _ in range(self._config_data.world_size)
            ]

            dist.all_gather(gathered, tensor)

            for rank, received_tensor in enumerate(gathered):
                gathered_by_rank[rank].append(received_tensor)

        if isinstance(message.payload, torch.Tensor):
            return [rank_tensors[0] for rank_tensors in gathered_by_rank]

        return gathered_by_rank
