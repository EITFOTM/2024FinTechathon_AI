import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from ModelDataset import *
from ModelTools import *
import torch.optim as optim
# import torch_directml
import time
# -i https://pypi.tuna.tsinghua.edu.cn/simple


def train(d: str = 'cpu',
          model_name: str = 'Cnn') -> None:
    """
    :param device: 在该设备上进行模型的训练
    :param model_name: 选择训练时所使用的模型
    :param class_number: 要求模型进行分类的个数，deepfake类型的问题类似与二分类问题，
    即分析输入图片的真假性。
    """
    # CUDA的深度神经网络库cuDNN可以进一步提升GPU对于神经网络的训练和推断过程
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 加载训练集
    print("Loading dataset...")
    root_dir = "data/Face2/Train"
    fake_dir = "Fake"
    real_dir = "Real"
    face3_root_dir = "data/Face3/Train"
    face3_fake_dir = "Fake"
    face3_real_dir = "Real"
    # 图像增强详见：https://blog.csdn.net/weixin_46334272/article/details/135395701
    # 进行数据预处理和增强操作:通过对训练集进行各种变换和扩增操作，可以增加训练数据的多样性和丰富性，从而提高模型的泛化能力。
    # 数据增强的目的是通过对训练集中的图像进行随机变换，生成更多样的图像样本，以模拟真实世界中的各种场景和变化。
    # 这样可以帮助模型学习到更多不同的特征和模式，提高模型对于输入数据的扰动、噪声、异常值等干扰的抵抗能力，同时减少过拟合的风险。
    # 常见的数据增强操作包括图像翻转、旋转、缩放、裁剪、平移、变换色彩空间、添加噪声等。增加样本的多样性，可以更好地训练出泛化能力强的模型
    trains_transforms = transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45度到45度之间
                                            transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转，10%的概率
                                            transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转，10%的概率
                                            # 参数分别为亮度，对比度，饱和度，色相，详细可参考：https://blog.csdn.net/lxhRichard/article/details/128083192
                                            transforms.ColorJitter(0.2, 0.1, 0.1, 0.1),
                                            transforms.RandomGrayscale(p=0.01),  # 随机转换成灰度率， 3通道就是R=G=B
                                            transforms.ToTensor()])
    fake_dataset = MyDataset(root_dir, fake_dir, transform=trains_transforms)
    real_dataset = MyDataset(root_dir, real_dir, transform=trains_transforms)
    face3_fake_dataset = MyDataset(face3_root_dir, face3_fake_dir, transform=trains_transforms)
    face3_real_dataset = MyDataset(face3_root_dir, face3_real_dir, transform=trains_transforms)
    train_dataset = fake_dataset + real_dataset + face3_real_dataset + face3_fake_dataset
    batch = 16
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # 开始训练模型
    print(f'You are training on device: {d}.')
    device = get_device(d)
    loss = nn.CrossEntropyLoss().to(device)
    model = model_create(model_name, device)
    # 优化器详解可参考：https://blog.csdn.net/2301_76846375/article/details/141476689
    # 随机梯度下降（SGD）是一种梯度下降形式，对于每次前向传递，都会从总的数据集中随机选择一批数据，即批次大小1。
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 基于余弦函数周期变化的学习率
    # Adam 优化器是一种自适应学习率的优化算法，结合了动量梯度下降和 RMSprop 算法的思想。
    # 它通过自适应地调整每个参数的学习率，从而在训练过程中加速收敛。
    # 所以Adam优化器的初始学习率可以随意设置，也不需要再添加周期变化的学习率
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 1  # 循环轮数
    optimizer_name = type(optimizer).__name__
    train_start_time = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{n_epochs}')
        for i, (images, labels) in enumerate(train_loader, 0):
            batch_start_time = time.time()
            train_loader_len = len(train_loader)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            result_loss = loss(outputs, labels)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            scheduler.step()
            current_loss = result_loss.item()
            running_loss += current_loss
            batch_end_time = time.time()

            # 更新进度条
            batch_num = i + 1
            update_progress_bar(batch_start_time, train_loader_len, batch_num, batch_end_time, current_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'\nLoss_mean: {running_loss / len(train_loader):.4f}, Total loss: {running_loss}.')
        minutes, seconds = divmod(epoch_time, 60)
        print(f'Total time spent on this epoch: {minutes}minutes {seconds}seconds')

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    minutes, seconds = divmod(train_time, 60)
    print(f'Train time: {minutes}minutes {seconds}seconds.')
    print("-" * 100)
    # 保存模型
    model_save(model_name, model, optimizer_name, n_epochs)


if __name__ == "__main__":
    time_start = time.time()
    # NVIDIA显卡用"cuda",其他显卡用"gpu",没有显卡用"cpu"
    train(d="gpu", model_name="efficientnet_b0")
    print("=" * 150)
    time_end = time.time()
    total_time = time_end - time_start
    minutes, seconds = divmod(total_time, 60)
    print(f'Total time : {minutes}minutes {seconds:.4f}seconds')
