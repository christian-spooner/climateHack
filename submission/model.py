import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.layer1 = nn.Conv3d(1, 4, kernel_size=5, stride=(1, 2, 2), padding=1)
        self.layer2 = nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.layer3 = nn.Conv3d(16, 16, kernel_size=3, stride=(1, 2, 2), padding=0)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 24 * 64 * 64)

    def forward(self, features):

        x = features.unsqueeze(dim=1) / 1023.0
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.pool(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.pool(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.pool(x))
        x = self.flatten(x)
        x = torch.relu(self.linear(x))
        x = x.view(-1, 24, 64, 64) * 1023.0

        a = features.shape[0]
        x2 = features[0:a, 11, 32:96, 32:96]
        x2 = torch.unsqueeze(x2, dim=1)
        y2 = []
        for i in range(24):
            y2.append(x2)
        x2 = torch.cat(y2, dim=1)

        z = (0.5 * x) + (0.5 * x2)

        return z
