import torch.backends.cudnn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from ModelDataset import *
from ModelTools import *
import torch.optim as optim
import time

# 图像增强详见：https://blog.csdn.net/weixin_46334272/article/details/135395701
# 进行数据预处理和增强操作:通过对训练集进行各种变换和扩增操作，可以增加训练数据的多样性和丰富性，从而提高模型的泛化能力。
# 数据增强的目的是通过对训练集中的图像进行随机变换，生成更多样的图像样本，以模拟真实世界中的各种场景和变化。
# 这样可以帮助模型学习到更多不同的特征和模式，提高模型对于输入数据的扰动、噪声、异常值等干扰的抵抗能力，同时减少过拟合的风险。
# 常见的数据增强操作包括图像翻转、旋转、缩放、裁剪、平移、变换色彩空间、添加噪声等。增加样本的多样性，可以更好地训练出泛化能力强的模型
transforms = {
    'Train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45度到45度之间
                                 transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转，10%的概率
                                 transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转，10%的概率
                                 # 参数分别为亮度，对比度，饱和度，色相，详细可参考：https://blog.csdn.net/lxhRichard/article/details/128083192
                                 transforms.ColorJitter(0.2, 0.1, 0.1, 0.1),
                                 transforms.RandomGrayscale(p=0.01),  # 随机转换成灰度率， 3通道就是R=G=B
                                 transforms.ToTensor()
                                 ]),
    'Valid': transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()
                                      ])
}
root_dir = "data"
datasets_dir = ["Face2", "Face3"]
classes = ["Fake", "Real"]
datasets = {'Train': [], 'Valid': []}
for i in datasets_dir:
    for j in ["Train", "Valid"]:
        path = os.path.join(root_dir, i, j)
        for k in classes:
            datasets[j].append(MyDataset(path, k, transform=transforms[j]))
batch = 32
image_datasets = {x: ConcatDataset(datasets[x]) for x in ['Train', 'Valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch, shuffle=True) for x in ['Train', 'Valid']}
print(dataloaders)
dataset_size = {x: image_datasets[x].__len__() for x in ['Train', 'Valid']}
print(dataset_size)
dataiter = iter(dataloaders['Valid'])
print(dataiter)
images, labels = next(dataiter)
print(images, labels)
columns = 2
rows = 1
fig = plt.figure(figsize=(20, 12))
def im_covert(tensor):
    image = tensor.to('cpu').clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


for idx in range(columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1)
    ax.set_title(labels[idx])
    plt.imshow(im_covert(images[idx]))
plt.show()
