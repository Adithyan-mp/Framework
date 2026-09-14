from communication.interface.base_communicator import BaseCommunicator
from config.loader.load_config import LoadConfig
import torch
import os
import torch.distributed as dist
from communication.interface.message import Message

class TorchCommunicator(BaseCommunicator):

    def __init__(self, fname: str | None = None):

        config_loader = LoadConfig(fname=fname)

        self.config_data = (
            config_loader.load_torch_communicator_config()
        )

        self._configure_environment()

    def _configure_environment(self):

        if self.config_data.nccl_socket_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = (
                self.config_data.nccl_socket_ifname
            )

        if self.config_data.gloo_socket_ifname:
            os.environ["GLOO_SOCKET_IFNAME"] = (
                self.config_data.gloo_socket_ifname
            )

        os.environ["MASTER_ADDR"] = (
            self.config_data.master_addr
        )

        os.environ["MASTER_PORT"] = str(
            self.config_data.master_port
        )

    def init(self):

        dist.init_process_group(
            backend=self.config_data.backend,
            rank=self.config_data.rank,
            world_size=self.config_data.world_size,
            timeout=self.config_data.timeout,
        )

    def terminate(self):

        if dist.is_initialized():
            dist.destroy_process_group()
            
    
    def send(self, message: Message) -> None:

        if message.destination is None:
            raise ValueError("Message destination is required")

        if message.payload is None:
            raise ValueError("Message payload is required")

        dist.send(
            tensor=message.payload,
            dst=message.destination
        )


    def recv(self, message: Message) -> Message:

        if message.source is None:
            raise ValueError("Message source is required")

        if message.payload is None:
            raise ValueError("Message payload buffer is required")

        dist.recv(
            tensor=message.payload,
            src=message.source
        )

        return message


    def broadcast(self, message: Message) -> None:

        if message.source is None:
            raise ValueError("Message source is required")

        if message.payload is None:
            raise ValueError("Message payload is required")

        dist.broadcast(
            tensor=message.payload,
            src=message.source
        )
        
    
    def all_gather(self, message : Message):
        if message.payload is None:
            raise ValueError("Message payload is required")
        
        gather_message = [torch.empty_like(message.payload) for _ in range(self.config_data.world_size)]
        
        dist.all_gather(gather_message,message.payload)
                
        return gather_message