import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from ModelDataset import *
from ModelTools import *
import torch.optim as optim
# import torch_directml
import time
# -i https://pypi.tuna.tsinghua.edu.cn/simple
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def train(device: str = 'cpu',
          model_name: str = 'Cnn') -> None:
    """
    :param device: 在该设备上进行模型的训练
    :param model_name: 选择训练时所使用的模型
    :param class_number: 要求模型进行分类的个数，deepfake类型的问题类似与二分类问题，
    即分析输入图片的真假性。
    """
    d = device
    if device in ['cuda', 'Cuda', 'CUDA']:
        device = torch.device(device)
    # elif device in ['gpu', 'GPU', 'Gpu']:
    #     device = torch_directml.device(0)
    else:
        device = torch.device('cpu')

    # 加载训练集
    root_dir = "data/Face2/Train"
    fake_dir = "Fake"
    real_dir = "Real"
    trains = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    fake_dataset = MyDataset(root_dir, fake_dir, transform=trains)
    real_dataset = MyDataset(root_dir, real_dir, transform=trains)
    train_dataset = fake_dataset + real_dataset
    batch = 16
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # 开始训练模型
    train_start_time = time.time()
    print(f'You are training on device: {d}.')
    loss = nn.CrossEntropyLoss().to(device)
    model = model_create(model_name, device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 基于余弦函数周期变化的学习率
    n_epochs = 1  # 循环轮数
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
            running_loss += result_loss.item()
            batch_end_time = time.time()

            # 更新进度条
            batch_num = i + 1
            update_progress_bar(batch_start_time, train_loader_len, batch_num, batch_end_time)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'\nLoss_mean: {running_loss / len(train_loader):.4f}, Total loss: {running_loss}.')
        print(f'Total time spent on this epoch: {epoch_time:.4f} seconds')

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f'Training time: {train_time:.4f} seconds')
    print("-" * 50)

    # 保存模型
    model_save(model_name, model)


if __name__ == "__main__":
    time_start = time.time()
    # NVIDIA显卡用"cuda",其他显卡用"gpu",没有显卡用"cpu"（不推荐用cpu，不然会很慢）
    train(device="cuda", model_name="efficientnet_b0")
    print("=" * 150)
    time_end = time.time()
    total_time = time_end - time_start
    print(f'Total training time : {total_time:.4f} seconds')
