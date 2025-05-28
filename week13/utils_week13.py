
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.functional.relu
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        h = self.conv1(x)  # [B, 6, 28, 28]
        h = self.pool(self.relu(h))  # [B, 6, 14, 14]
        h = self.conv2(h)  # [B, 16, 10, 10]
        h = self.pool(self.relu(h))  # [B, 16, 5, 5]
        h = h.reshape(x.shape[0], -1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        o = self.fc3(h)
        return o