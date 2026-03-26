import os
import time

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DummyDataset(Dataset):
    def __init__(self, size=500):
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 3))
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Batch Size: {batch_size}, Device: {device}")
    print(f"MLflow URI: {mlflow_uri}")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("ml-energy-poc")

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("device", str(device))

        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(DummyDataset(500), batch_size=batch_size, shuffle=True)

        start = time.time()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("loss", float(loss.item()), step=step)
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        duration = time.time() - start
        mlflow.log_metric("duration_seconds", duration)
        print(f"Fertig! Dauer: {duration:.2f}s")


if __name__ == "__main__":
    main()
