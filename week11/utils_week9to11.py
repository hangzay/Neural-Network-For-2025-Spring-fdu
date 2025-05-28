import numpy as np
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

def train(model, loader, criterion, optimizer, device=None):
    device = next(model.parameters()).device if device is None else device
    model.train()
    train_loss = 0.0
    sample_num = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sample_num += inputs.shape[0]
        train_loss += loss.item() * inputs.shape[0]
    train_loss /= sample_num
    return train_loss

def test(model, loader, device=None):
    device = next(model.parameters()).device if device is None else device
    model.eval()
    total_num = 0.
    correct_num = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total_num += inputs.shape[0]
            correct_num += predicted.eq(labels).sum().item()
    acc = correct_num / total_num * 100.
    return acc