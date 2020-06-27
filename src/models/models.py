import torch

import torch.nn as nn
import torch.nn.functional as F
import os.path as path

ROOT_DIR = path.abspath(path.join(__file__ ,"../../.."))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        x = torch.randn(128, 173).view(-1, 1, 128, 173)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 4)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        if self._to_linear is None:
            # find the number of the input of the linear layer
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


class ComplexCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3)

        x = torch.randn(128, 173).view(-1, 1, 128, 173)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 4)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        # max pooling over 2x2
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, 3)
        self.conv8 = nn.Conv2d(256, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, 3)

        x = torch.randn(128, 173).view(-1, 1, 128, 173)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 4)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        # max pooling over 2x2
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.relu(self.bn3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.relu(self.bn4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv9(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

