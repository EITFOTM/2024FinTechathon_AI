import torch
from torchvision import transforms

from ModelEfficientNet import efficientnet_b0
from ModelCnn import *
from ModelDataset import *
import time
import torch.optim as optim
from torch.utils.data import DataLoader
# import torch_directml


def train(device: str):
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
    # model = Cnn().to(device)
    model = efficientnet_b0(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    n_epochs = 1  # 循环轮数
    progress_bar_length = 25  # 进度条长度
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{n_epochs}')
        for i, (images, labels) in enumerate(train_loader, 0):
            batch_start_time = time.time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            result_loss = loss(outputs, labels)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += result_loss.item()

            # 更新进度条
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_num = i + 1
            progress = batch_num / len(train_loader)
            progress_bar = ('>' * int(progress * progress_bar_length) +
                            '-' * (progress_bar_length - int(progress * progress_bar_length)))
            # 在同一行显示进度条
            print(
                f'\r\tBatch {i + 1}/{len(train_loader)} [{progress_bar}] Loss: {result_loss.item():.4f}, Time: {batch_time:.4f} seconds', end="")

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'\nLoss_mean: {running_loss / len(train_loader):.4f}, Total loss: {running_loss}.')
        print(f'Total time spent on this epoch: {epoch_time:.4f} seconds')
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f'Training time: {train_time:.4f} seconds')
    print("-" * 50)
    # 保存模型
    torch.save(model.state_dict(), 'ModelCnn.pt')
    # 加载模型
    # print(torch.load('model_Cnn.pt'))
    # from ModelCnn import Cnn
    # model_cnn = Cnn.load_state_dict(torch.load('ModelCnn1.pt'))
    # 加载测试集
    test_root_dir = "data/Face2/Train"
    test_fake_dir = "Fake"
    test_real_dir = "Real"
    tests = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_fake_dataset = MyDataset(test_root_dir, test_fake_dir, transform=tests)
    test_real_dataset = MyDataset(test_root_dir, test_real_dir, transform=tests)
    test_dataset = test_fake_dataset + test_real_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    # 评估模型
    correct = 0
    total = 0
    print(f'You are testing on device: {d}.')
    test_start_time = time.time()
    with torch.no_grad():
        for j, (test_images, test_labels) in enumerate(test_loader, 0):
            batch_start_time = time.time()
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
            # 更新进度条
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            test_batch_num = j + 1
            test_progress = test_batch_num / len(test_loader)
            test_progress_bar = ('>' * int(test_progress * progress_bar_length) +
                            '-' * (progress_bar_length - int(test_progress * progress_bar_length)))
            # 在同一行显示进度条
            print(
                f'\r\tBatch {j + 1}/{len(test_loader)} [{test_progress_bar}], Time: {batch_time:.4f} seconds',
                end="")
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f'\nAccuracy:{100 * correct / total:.2f}%,on {d}, Test Time: {test_time:.4f} seconds.')


if __name__ == "__main__":
    time_start = time.time()
    # NVIDIA显卡用"cuda",其他显卡用"gpu",没有显卡用"cpu"（不推荐用cpu，不然会很慢）
    train(device="cuda")
    print("=" * 150)
    time_end = time.time()
    total_time = time_end - time_start
    print(f'Total time : {total_time:.4f} seconds')
