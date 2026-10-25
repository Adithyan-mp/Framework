from communication.interface.base_topology import BaseTopology
from config.loader.load_config import LoadConfig
class TreeTopology(BaseTopology):

    def __init__(
        self,
        config_loader:LoadConfig
    ):
        self.config_data = config_loader.load_tree_topology()
        self.topology = self.config_data.data
        self.root =  self.config_data.root

    def get_neighbours(self, node: int) -> list[int]:
        neighbours = self.topology.get(node, []).copy()

        parent = self.get_parent(node)

        if parent is not None:
            neighbours.append(parent)

        return neighbours

    def is_connected(self, node1: int, node2: int) -> bool:
        return (
            node2 in self.topology.get(node1, [])
            or node1 in self.topology.get(node2, [])
        )

    def get_parent(self, node: int) -> int | None:
        for parent, children in self.topology.items():
            if node in children:
                return parent

        return None

    def get_children(self, node: int) -> list[int]:
        return self.topology.get(node, []).copy()