from dataclasses import dataclass
from datetime import timedelta


@dataclass
class TorchCommunicatorConfig:
    rank: int
    world_size: int
    backend: str
    master_addr: str
    master_port: int

    timeout: timedelta | None = None
    nccl_socket_ifname: str | None = None
    gloo_socket_ifname: str | None = None