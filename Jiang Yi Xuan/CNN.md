# 搭建 CNN 模型技术文档

### 初始化方法

在 `init` 方法中，定义了网络的所有层。

**卷积层:**

```python
# 卷积层，用于提取图像特征  
self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))  
self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))  
self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))  
self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
```

- `self.conv1`: 第一个卷积层，接收 3 个输入通道（对应于 RGB 图像的三个颜色通道），输出 64 个通道，使用 3x3 的卷积核，步长为 1，填充为 1 以保持输入和输出尺寸相同。
- `self.conv2`: 第二个卷积层，接收 64 个输入通道，输出 128 个通道，同样使用 3x3 的卷积核，步长为 1，填充为 1。
- `self.conv3`: 第三个卷积层，将通道数从 128 减少到 64，使用 3x3 的卷积核，步长为 1，填充为 1。
- `self.conv4`: 第四个卷积层，将通道数从 64 减少到 32，使用 3x3 的卷积核，步长为 1，填充为 1。

**池化层:**

```python
# 池化层，用于减少特征图的尺寸  
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
```

- `self.pool1`, `self.pool2`, `self.pool3`, `self.pool4`: 每个卷积层后面都跟着一个最大池化层，使用 2x2 的池化核，步长为 2，没有填充。这些层用于减少特征图的尺寸，增加特征的抽象程度。

**展平层:**

```python
# 展平层，将多维特征图展平为一维  
self.flat = nn.Flatten()
```

- `self.flat`: 将多维的特征图展平成一维，以便输入到全连接层。

**全连接层:**

```python
# 全连接层，将特征映射到指定维度  
self.fc1 = nn.Linear(6272, 784)  
self.fc2 = nn.Linear(784, 98)  
self.fc3 = nn.Linear(98, 2)
```

- `self.fc1`, `self.fc2`, `self.fc3`: 三个全连接层，分别将特征从 6272 维映射到 784 维，784 维映射到 98 维，最后 98 维映射到 2 维（假设是二分类任务）。

**激活函数和 Dropout:**

```python
# 激活函数和Dropout层  
self.softmax = nn.LogSoftmax(dim=1)  
self.relu = nn.ReLU()  
self.dropout = nn.Dropout(0.25)
```

- `self.softmax`, `self.relu`: 激活函数，分别用于输出层的对数 softmax 激活和隐藏层的 ReLU 激活。
- `self.dropout`: Dropout 层，用于防止过拟合，这里设置了 0.25 的 dropout 率。

### 前向传播方法

在 `forward` 方法中，定义了数据通过网络的前向传播路径。

```python
def forward(self, x):  
    x = self.pool1(self.relu(self.conv1(x)))  
    x = self.pool2(self.relu(self.conv2(x)))  
    x = self.pool3(self.relu(self.conv3(x)))  
    x = self.pool4(self.relu(self.conv4(x)))  
    x = self.flat(x)  
    x = self.relu(self.fc1(x))  
    x = self.dropout(x)  
    x = self.fc2(x)  
    x = self.dropout(x)  
    x = self.fc3(x)  
    x = self.softmax(x)  
    return x
```

- 数据通过每个卷积层后，先进行 ReLU 激活，再进行最大池化。
- 展平后，数据通过全连接层，中间使用 ReLU 激活和 Dropout。
- 最后，通过 softmax 层得到输出。
