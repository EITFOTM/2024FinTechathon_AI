from ModelDataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ModelTools import *
import time


def test(normalize,
         d: str,
         pre_epochs: int,
         model_name: str = 'Cnn',
         optimizer_name: str = "SGD") -> None:
    """
    :param d: 在该设备上进行模型的测试
    :param model_name: 选择测试时所使用的模型，需要该模型已经训练完成
    """

    # 加载测试集
    print("Loading dataset...")
    root_dir = "data"
    datasets_dir = ["Face2", "Face3"]
    data_type = ["Test"]
    classes = ["Fake", "Real"]
    phase = "Test"
    batch = 96
    data_transforms = {'Test': transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor()])
                       }
    loaders, dataset_size = datasets_load(root_dir, datasets_dir, data_type, classes, batch, data_transforms)
    test_loader = loaders[phase]

    # 加载模型
    device = get_device(d)
    print(f'You are testing on the {device}. Batch size is {batch} and optimizer name is {optimizer_name}.\n'
          f'Model name is {model_name}.')

    model, best_acc, optimizer_state_dict, history = model_load(model_name, device, d, optimizer_name, pre_epochs)
    # model = globals()[model_name]()  # 找到对应模型并调用它
    # model.to(device)
    # model_path = f'Model{model_name}.pt'
    # model.load_state_dict(torch.load(model_path, weights_only=False))
    # print(f'You are testing on device: {d}.')

    # 测试模型
    correct = 0
    total = 0
    y_true = []
    y_pred = []
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
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(test_labels.cpu().numpy())
            correct += (predicted == test_labels).sum().item()
            batch_end_time = time.time()

            # 更新进度条
            batch_num = i + 1
            update_progress_bar(phase, batch_start_time, test_loader_len, batch_num, batch_end_time)

    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print('\n')
    # 混淆矩阵分析模型预测的结果
    title = f'Model{model_name}_{optimizer_name}_e{pre_epochs}'
    cm = ConfusionMatrix(y_true, y_pred, classes, normalize, title)
    # 混淆矩阵可视化
    cm.plot_confusion_matrix()
    # 模型参数量/计算量和推理速度计算
    evaluation(model, device)
    minutes, seconds = divmod(test_time, 60)
    print(f'Total time: {minutes:.0f} minutes {seconds:.1f} seconds.')
    print(f'Total accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    pre_epochs = 3
    optimizer_name = 'Adam'
    model_name = "efficientnet_b0"
    d = 'cuda'
    normalize = True
    # NVIDIA显卡用"cuda"，没有显卡用"cpu"
    test(d=d, pre_epochs=pre_epochs, model_name=model_name, optimizer_name=optimizer_name,normalize=normalize)
    print("=" * 150)
