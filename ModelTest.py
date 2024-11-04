# import torch_directml
from ModelDataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ModelTools import *
import time


def test(device: str,
         model_name: str = 'Cnn') -> None:
    """
    :param device: 在该设备上进行模型的测试
    :param model_name: 选择测试时所使用的模型，需要该模型已经训练完成
    """
    d = device
    if device in ['cuda', 'Cuda', 'CUDA']:
        device = torch.device(device)
    # elif device in ['gpu', 'GPU', 'Gpu']:
    #     device = torch_directml.device(0)
    else:
        device = torch.device('cpu')

    # 加载模型
    model = model_load(model_name, device)

    # 加载测试集
    test_root_dir = "data/Face2/Train"
    test_fake_dir = "Fake"
    test_real_dir = "Real"
    batch = 16
    tests = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_fake_dataset = MyDataset(test_root_dir, test_fake_dir, transform=tests)
    test_real_dataset = MyDataset(test_root_dir, test_real_dir, transform=tests)
    test_dataset = test_fake_dataset + test_real_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    # 测试模型
    correct = 0
    total = 0
    progress_bar_length = 25
    print(f'You are testing on device: {d}.')
    test_start_time = time.time()
    with torch.no_grad():
        for i, (test_images, test_labels) in enumerate(test_loader, 0):
            batch_start_time = time.time()
            test_loader_len = len(test_loader)
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
            # print(predicted)
            # print(test_labels)
            # print(correct)
            batch_end_time = time.time()

            # 更新进度条
            batch_num = i + 1
            update_progress_bar(batch_start_time, test_loader_len, batch_num, batch_end_time)

    # 评估模型训练效果
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f'\nAccuracy:{100 * correct / total:.2f}%,on {d}, Test Time: {test_time:.4f} seconds.')


if __name__ == '__main__':
    time_start = time.time()
    # NVIDIA显卡用"cuda",其他显卡用"gpu",没有显卡用"cpu"（不推荐用cpu，不然会很慢）
    test(device='cuda', model_name='efficientnet_b0')
    print("=" * 150)
    time_end = time.time()
    total_time = time_end - time_start
    print(f'Total time : {total_time:.4f} seconds')
