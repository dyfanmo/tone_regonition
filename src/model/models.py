import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """ The base model """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)

        x = torch.randn(128, 87).view(-1, 1, 128, 87)
        self._to_linear = None
        self.convs(x)

        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_a = nn.Linear(512, 5)
        self.fc2_b = nn.Linear(512, 6)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        if self._to_linear is None:
            # find the number of the input of the linear layer
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = self.dropout2(F.relu(self.fc1(x)))
        x1 = self.fc2_a(x)
        x2 = self.fc2_b(x)
        yhat1 = self.log_softmax(x1)
        yhat2 = self.log_softmax(x2)

        return yhat1, yhat2



class ComplexCNN(nn.Module):
    """ A model more complex than the SimpleCNN bu only increasing the hidden units """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)

        x = torch.randn(128, 87).view(-1, 1, 128, 87)
        self._to_linear = None
        self.convs(x)

        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_a = nn.Linear(512, 5)
        self.fc2_b = nn.Linear(512, 6)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = self.dropout2(F.relu(self.fc1(x)))
        x1 = self.fc2_a(x)
        x2 = self.fc2_b(x)
        yhat1 = self.log_softmax(x1)
        yhat2 = self.log_softmax(x2)

        return yhat1, yhat2


class DeepCNN(nn.Module):
    """ A deep CNN """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, 3)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, 3)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, 2)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, 2)

        x = torch.randn(128, 87).view(-1, 1, 128, 87)
        self._to_linear = None
        self.convs(x)

        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_a = nn.Linear(512, 5)
        self.fc2_b = nn.Linear(512, 6)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def convs(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.conv12(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = self.dropout2(F.relu(self.fc1(x)))
        x1 = self.fc2_a(x)
        x2 = self.fc2_b(x)
        yhat1 = self.log_softmax(x1)
        yhat2 = self.log_softmax(x2)

        return yhat1, yhat2