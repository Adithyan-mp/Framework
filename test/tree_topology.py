from communication.topology.tree_topology import TreeTopology


topology_data = {
    0: [1, 2, 3],
    1: [],
    2: [4, 5],
    3: [6],
    6: [7],
    7: [8, 9],
    4: [],
    5: [],
    8: [],
    9: []
}

topology = TreeTopology(topology=topology_data)

topology = TreeTopology(topology=topology_data, root=0)

assert set(topology.get_neighbours(0)) == {1, 2, 3}
assert set(topology.get_neighbours(7)) == {6, 8, 9}

assert topology.get_parent(0) is None
assert topology.get_parent(7) == 6
assert topology.get_parent(9) == 7

assert topology.get_children(0) == [1, 2, 3]
assert topology.get_children(7) == [8, 9]
assert topology.get_children(9) == []

assert topology.is_connected(0, 1) is True
assert topology.is_connected(1, 0) is True
assert topology.is_connected(7, 8) is True
assert topology.is_connected(8, 7) is True

assert topology.is_connected(1, 2) is False
assert topology.is_connected(4, 5) is False
assert topology.is_connected(1, 6) is False
assert topology.is_connected(0, 0) is False

assert topology.get_neighbours(100) == []
assert topology.get_children(100) == []
assert topology.get_parent(100) is None