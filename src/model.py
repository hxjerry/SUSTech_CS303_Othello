import torch
from typing import Tuple
import numpy as np

CHANNELS = 64

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FCBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(FCBlock, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OthelloNet(torch.nn.Module):
    unpacker_mask = torch.tensor([1 << i for i in range(64)], dtype=torch.uint64).view(1, 1, 8, 8)
    def to_tensor(self, player: np.uint64, opponent: np.uint64) -> torch.Tensor:
        board = ((torch.from_numpy(np.array([player, opponent], dtype=np.uint64)).view(1, 2, 1, 1) & self.unpacker_mask) != 0).float()
        return board

    def __init__(self):
        super(OthelloNet, self).__init__()
        self.conv1 = ConvBlock(2, CHANNELS, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(CHANNELS, CHANNELS, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(CHANNELS, CHANNELS, kernel_size=3, padding=0)
        self.conv4 = ConvBlock(CHANNELS, CHANNELS, kernel_size=3, padding=0)

        self.fc1 = FCBlock(CHANNELS * 4 * 4, 256)
        self.fc_policy = torch.nn.Linear(256, 64)
        self.fc_value = torch.nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))

        return policy, value

if __name__ == "__main__":

    model = OthelloNet()
    sample_player = np.uint64(0x0000000810000000)
    sample_opponent = np.uint64(0x0000001008000000)
    input_tensor = model.to_tensor(sample_player, sample_opponent)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    input_tensor =input_tensor.to(device)
    model.eval()
    model.compile(mode="max-autotune")

    with torch.inference_mode():
        model(input_tensor)  # Warm-up

    import time
    start_time = time.time()
    with torch.inference_mode():
        for _ in range(100000):
            policy, value = model(input_tensor)
    end_time = time.time()
    print(f"Average throughput: {100000 / (end_time - start_time)} inferences per second")