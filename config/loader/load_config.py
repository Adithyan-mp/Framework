from config.schema.communicator_config import TorchCommunicatorConfig

import yaml
from datetime import timedelta


class LoadConfig:

    def __init__(self, fname: str | None = None):

        if not fname:
            raise RuntimeError("Enter a valid config file path")

        with open(fname, "r") as f:
            self.config_data = yaml.safe_load(f)

    def load_torch_communicator_config(self):

        communication = self.config_data["communication"]

        return TorchCommunicatorConfig(
            rank=communication["rank"],
            world_size=communication["world_size"],
            master_addr=communication["master_addr"],
            master_port=communication["master_port"],
            backend=communication["backend"],

            gloo_socket_ifname=communication.get(
                "gloo_socket_ifname"
            ),

            nccl_socket_ifname=communication.get(
                "nccl_socket_ifname"
            ),

            timeout=(
                timedelta(seconds=communication["timeout"])
                if communication.get("timeout") is not None
                else None
            ),
        )