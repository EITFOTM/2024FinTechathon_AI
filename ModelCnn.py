from torch import nn


class Cnn(nn.Module):
    def __init__(self,
                 num_classes: int = 2):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(6272, 784)
        self.fc2 = nn.Linear(784, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
