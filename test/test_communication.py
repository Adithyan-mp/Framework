from communication.communicator.torch_communicator import TorchCommunicator
from communication.interface.message import Message

import torch


def send_recv_test():

    communicator = TorchCommunicator("./config.yaml")

    communicator.init()

    rank = communicator.get_rank()

    payload = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ])

    if rank == 0:

        message = Message(
            message_type="weight",
            source=rank,
            destination=1,
            payload=payload
        )

        communicator.send(message)

        print(
            f"Rank {rank}: message sent\n"
            f"{message.payload}"
        )

    elif rank == 1:

        temp_t = torch.zeros_like(payload)

        message = Message(
            message_type="weight",
            source=0,
            destination=rank,
            payload=temp_t
        )

        communicator.recv(message)

        print(
            f"Rank {rank}: message received\n"
            f"{message.payload}"
        )

    communicator.terminate()

def send_recv_test2():

    communicator = TorchCommunicator("./config.yaml")

    communicator.init()

    rank = communicator.get_rank()
    world_size = communicator.get_world_size()

    payload = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

    if rank == 0:

        for i in range(1, world_size):

            message = Message(
                message_type="weight",
                source=rank,
                destination=i,
                payload=payload
            )

            communicator.send(message)

            print(
                f"Rank {rank}: message sent to {i}\n"
                f"{message.payload}"
            )

    else:

        temp_t = torch.zeros_like(payload)

        message = Message(
            message_type="weight",
            source=0,
            destination=rank,
            payload=temp_t
        )

        communicator.recv(message)

        print(
            f"Rank {rank}: message received\n"
            f"{message.payload}"
        )

    communicator.terminate()
            
def broadcast_test():

    communicator = TorchCommunicator("./config.yaml")

    communicator.init()

    rank = communicator.get_rank()

    if rank == 0:

        payload = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])

    else:

        payload = torch.zeros(
            (2, 3),
            dtype=torch.int64
        )

    message = Message(
        message_type="weight",
        source=0,
        payload=payload
    )

    communicator.broadcast(message)

    print(
        f"Rank {rank}: broadcast result\n"
        f"{message.payload}"
    )

    communicator.terminate()
    
def all_gather_test():

    communicator = TorchCommunicator("./config.yaml")

    communicator.init()

    rank = communicator.get_rank()

    payload = torch.tensor([
        [rank, rank, rank],
        [rank, rank, rank]
    ])

    message = Message(
        message_type="weight",
        source=rank,
        payload=payload
    )

    gathered_msg = communicator.all_gather(message)

    print(
        f"Rank {rank}: all gather result\n"
        f"{gathered_msg}"
    )

    communicator.terminate()
    
def invalid_destination_test():
    communicator = TorchCommunicator('./config.yaml')
    communicator.init()
    message = Message(message_type="weight",
                      payload=torch.tensor([1,2,3]),
                      destination=None)
    
    try:
        communicator.send(message=message)
    except ValueError as e:
        print(f"Expected error : {e}")
    
    communicator.terminate()
    
def invalid_source_test():
    communicator = TorchCommunicator("./config.yaml")
    communicator.init()

    message = Message(
        message_type="weight",
        payload=torch.zeros(3),
        source=None
    )

    try:
        communicator.recv(message)
    except ValueError as e:
        print(f"Expected error: {e}")

    communicator.terminate()

def invalid_payload_test():
    communicator = TorchCommunicator("./config.yaml")
    communicator.init()
    
    message = Message(message_type="weight",destination=1,payload=None)
    
    try:
        communicator.send(message=message)
    except ValueError as e:
        print(f"Expected error : {e}")
    
    communicator.terminate()
    
def gpu_send_recv_test():

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU test.")
        return

    communicator = TorchCommunicator("./config.yaml")
    communicator.init()

    rank = communicator.get_rank()

    device = torch.device("cuda")

    if rank == 0:

        payload = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            device=device
        )

        message = Message(
            message_type="weight",
            destination=1,
            payload=payload
        )

        communicator.send(message)

        print(
            f"Rank {rank}: GPU message sent\n"
            f"{message.payload}"
        )

    elif rank == 1:

        payload = torch.zeros(
            (2, 3),
            dtype=torch.int64,
            device=device
        )

        message = Message(
            message_type="weight",
            source=0,
            payload=payload
        )

        communicator.recv(message)

        expected = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            device=device
        )

        assert torch.equal(message.payload, expected)

        print(
            f"Rank {rank}: GPU message received\n"
            f"{message.payload}"
        )

    communicator.terminate()
    
def gpu_broadcast_test():

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU test.")
        return

    communicator = TorchCommunicator("./config.yaml")
    communicator.init()

    rank = communicator.get_rank()

    device = torch.device("cuda")

    if rank == 0:

        payload = torch.tensor(
            [
                [10, 20, 30],
                [40, 50, 60]
            ],
            device=device
        )

    else:

        payload = torch.zeros(
            (2, 3),
            dtype=torch.int64,
            device=device
        )

    message = Message(
        message_type="weight",
        source=0,
        payload=payload
    )

    communicator.broadcast(message)

    expected = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60]
        ],
        device=device
    )

    assert torch.equal(message.payload, expected)

    print(
        f"Rank {rank}: GPU broadcast result\n"
        f"{message.payload}"
    )

    communicator.terminate()
    

def gpu_all_gather_test():

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU test.")
        return

    communicator = TorchCommunicator("./config.yaml")
    communicator.init()

    rank = communicator.get_rank()
    world_size = communicator.get_world_size()

    device = torch.device("cuda")

    payload = torch.tensor(
        [rank],
        dtype=torch.int64,
        device=device
    )

    message = Message(
        message_type="weight",
        source=rank,
        payload=payload
    )

    gathered_msg = communicator.all_gather(message)

    assert len(gathered_msg) == world_size

    for i, tensor in enumerate(gathered_msg):
        expected = torch.tensor(
            [i],
            dtype=torch.int64,
            device=device
        )

        assert torch.equal(tensor, expected)

    print(
        f"Rank {rank}: GPU all-gather result\n"
        f"{gathered_msg}"
    )

    communicator.terminate()
    
if __name__ == "__main__":
    broadcast_test()
    all_gather_test()
    invalid_destination_test()