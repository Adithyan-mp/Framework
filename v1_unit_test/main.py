from computation.interface.base_dataset import BaseDataset
from computation.interface.base_model import BaseModel
from coordination.tree.fed_tree import FedTree

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MyModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 3)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class MyData(BaseDataset):
    def __init__(self, path):
        super().__init__()

        df = pd.read_csv(path)

        # Remove identifier if present
        df = df.drop(columns=["Id"], errors="ignore")
        df = df.dropna().drop_duplicates().reset_index(drop=True)

        # Separate features and labels
        X = df.drop(columns=["Species"]).to_numpy(dtype="float32")
        y = df["Species"]

       
        self.label_encoder = None
        y = y.to_numpy()

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to tensors
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    dataset = MyData("./v1_unit_test/Iris_c.csv")

    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
    )

    model = MyModel()

    # Check the model and dataset before running FedTree
    features, labels = next(iter(loader))
    outputs = model(features)

    print("Dataset size:", len(dataset))
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print("Model output shape:", outputs.shape)

    coordinator = FedTree(
        model=model,
        config="./v1_unit_test/config.yamal",
    )

    coordinator.run(dataloader=loader)