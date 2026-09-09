from dataclasses import dataclass
from datetime import timedelta

@dataclass
class TorchCommunicatorConfig:
    rank : int
    world_size : int
    backend : str
    timeout : timedelta | None = None
    
    master_port : str
    master_addr : str
    nccl_socket_ifname: str | None = None
    gloo_socket_ifname: str | None = None
    