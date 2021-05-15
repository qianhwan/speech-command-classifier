import torch
import torch.nn as nn
import torch.nn.functional as F


# copied and modified from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html#define-the-network  # noqa
class Model(nn.Module):
    def __init__(self, n_input=1, n_output=30, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.conv2_drop(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
