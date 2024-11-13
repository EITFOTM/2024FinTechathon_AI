import thop
import torch
import cpuinfo
import torchvision
import numpy as np
from ModelCnn import *
from ModelDataset import *
from ModelEfficientNet import *
import sklearn.metrics as metrics
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
# pip install py-cpuinfo -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install thop -i https://pypi.tuna.tsinghua.edu.cn/simple


def get_device(device: str):
    """

    :param device: 训练所使用的设备名字
    :return: 找到该设备输，并输出该设备信息
    """
    # 获取CPU信息
    info = cpuinfo.get_cpu_info()
    # 打印结果
    print("+" + 100 * "-" + "+")
    print(f"CPU version: {info['brand_raw']:<28}")
    print("{:<8}\t{:<5}\t{:<10}\t{:<8}".format("Architecture", "Bits", "Frequency", "Thread count"))
    print("{:<13}\t{:<4}\t{:<13}\t{:<8}"
          .format(info['arch'], info['bits'], info['hz_actual_friendly'], info['count']))
    print("+" + 100 * "-" + "+")
    if device in ['cuda', 'Cuda', 'CUDA'] and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU version: {properties.name:<28}")
        print("{:<8}\t{:<18}\t{:<10}\t".format("Total memory", "CUDA capability", "Multiprocessor count"))
        memory = str(properties.total_memory / 1024 / 1024) + 'MB'
        capa = str(properties.major)+'.'+str(properties.minor)
        print("{:<18}\t{:<20}\t{:<10}\t"
              .format(memory, capa, properties.multi_processor_count))
        print("+" + 100 * "-" + "+")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def datasets_load(root_dir: str, datasets_dir: list, data_type: list,
                  classes: list, batch: int, data_transforms: dict,):
    """

    :param root_dir: 数据集存放的根目录
    :param datasets_dir: 训练所使用的数据集列表
    :param data_type: 数据的使用类型，用于训练、验证还是测试
    :param classes: 标签列表
    :param batch: 每一批数据的大小
    :param data_transforms: 数据所进行的预处理操作
    :return: 返回经过预处理后和加载后的数据集以及数据集的大小
    """
    datasets = {}
    for t in data_type:
        datasets[t] = []

    for i in datasets_dir:
        for j in data_type:
            path = os.path.join(root_dir, i, j)
            for k in classes:
                datasets[j].append(MyDataset(path, k, transform=data_transforms[j]))
    image_datasets = {x: ConcatDataset(datasets[x]) for x in data_type}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch, shuffle=True) for x in data_type}
    dataset_size = {x: image_datasets[x].__len__() for x in data_type}
    return dataloaders, dataset_size


def update_progress_bar(phase: str, start_time: float, loader_len: int,
                        batch_num: int, end_time: float, current_loss: float = None):
    """

    :param phase: 当前运行所处的阶段（训练、验证、测试）
    :param start_time: 该阶段处理Batch开始的时间
    :param loader_len: 该阶段总Batch的总数
    :param batch_num: 该阶段处理完Batch的个数
    :param end_time: 该阶段处理Batch结束的时间
    :param current_loss: 当前Batch的损失
    :return: 打印该阶段的进度条，无返回值
    """
    # 更新进度条
    if current_loss is None:
        current_loss = 0
    progress_bar_length = 25  # 进度条长度
    batch_time = end_time - start_time
    progress = batch_num / loader_len
    progress_bar = ('>' * int(progress * progress_bar_length) +
                    '-' * (progress_bar_length - int(progress * progress_bar_length)))
    # 在同一行显示进度条
    print(
        f'\r\t{phase} batch {batch_num}/{loader_len} [{progress_bar}], '
        f'Time: {batch_time:.4f} seconds, '
        f'Current loss: {current_loss:.4f}', end="")


def model_load(model_name: str, device: torch.device, optimizer_name: str, epochs: int):
    """

    :param model_name: 训练时所使用的模型名字
    :param device: 使用的设备
    :param optimizer_name: 训练时所使用的优化器名字
    :param epochs: 训练的轮次数
    :return: 返回之前训练完的模型、准确率和优化器参数
    """
    model = globals()[model_name]()  # 找到对应模型并调用它
    model.to(device)
    model_path = f'Model{model_name}_{optimizer_name}_e{epochs}.pt'
    state = torch.load(model_path, weights_only=False)
    model.load_state_dict(state['state_dict'])
    best_acc = state['best_acc']
    optimizer_state_dict = state['optimizer']
    return model, best_acc, optimizer_state_dict


def model_save(model_name: str, state: dict, optimizer_name: str, epochs: int):
    """

    :param model_name: 训练所使用的模型名字
    :param state: 模型的结构、准确率和
    :param optimizer_name: 训练时所使用的优化器名字
    :param epochs: 训练的轮次数
    :return: 保存模型，无返回值
    """
    model_path = f'Model{model_name}_{optimizer_name}_e{epochs}.pt'
    torch.save(state, model_path)


def model_create(model_name: str, device: torch.device):
    """

    :param model_name: 所使用的模型名字
    :param device: 使用的设备
    :return: 返回创建完的模型
    """
    model = globals()[model_name]()
    model.to(device)
    return model


def evaluation(model, device):
    """
    深度学习模型参数量/计算量和推理速度计算
    https://zhuanlan.zhihu.com/p/376925457?utm_id=0
    :param model: 要评估的模型
    :param device: 所使用的设备
    :return: 打印模型的评估结果，无返回值
    """
    # FLOPs和Params计算
    optimal_batch_size = 16
    inputs = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(device)
    flops, params = thop.profile(model, inputs=(inputs,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))

    # 模型推理速度正确计算
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(inputs)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(f' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}')
    print('mean_syn:', mean_syn)

    # 模型吞吐量计算
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitions * optimal_batch_size) / total_time
    print('FinalThroughput:', throughput)


class ConfusionMatrix:
    """
    使用Python绘制混淆矩阵
    https://blog.csdn.net/xiaoshiqi17/article/details/136074424
    """
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.matrix = metrics.confusion_matrix(self.y_true, self.y_pred)

        self.get_confusion_matrix()

    def get_confusion_matrix(self):
        total_num = len(self.y_true)
        matrix = self.matrix
        tp = matrix[0][0]
        fp = matrix[0][1]
        fn = matrix[1][0]
        tn = matrix[1][1]
        accuracy = 100 * (tp + tn)/total_num
        ppv = 100 * tp/(tp+fp)
        tpr = 100 * tp/(tp+fn)
        tnr = 100 * tn/(tn+fn)
        print(matrix)
        cm_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)
        print(f'Accuracy:{accuracy:.2f}%')
        print(f'PPV:{ppv:.2f}%')
        print(f'TPR:{tpr:.2f}%')
        print(f'TNR:{tnr:.2f}%')

    def plot_confusion_matrix(self):
        pass


if __name__ == '__main__':
    device = get_device('cuda')
    print(device)
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torchvision.__version__)
    # optimizer_name = "Adam"
    # epochs = "3"
    # model = model_load('efficientnet_b0', device, optimizer_name, epochs)
    # model_name = 'efficientnet_b0'

    # model = globals()[model_name]()  # 找到对应模型并调用它
    # model.to(device)
    # model_path = 'Modelefficientnet_b0_Adam_e3.pt'
    # l = torch.load(model_path, weights_only=False)
    # best_acc = l['best_acc'].item()

    # print(l['s'])
    # model = globals()[model_name]()  # 找到对应模型并调用它
    # model.to(device)
    # model_path = f'Model{model_name}_{optimizer_name}_e{epochs}.pt'
    # state = torch.load(model_path, weights_only=False)
    # model.load_state_dict(state['state_dict'])
    # best_acc = state['best_acc']
    # optimizer_state_dict = state['optimizer']
    # return model, best_acc, optimizer_state_dict
    # evaluation(model, device)
    # true = torch.tensor([1,1,0,0,1,0,1,0,1]).to(device)
    # pred = torch.tensor([1,1,1,1,1,0,0,0,0]).to(device)
    # print(true.device.index)
    # print(type(true))
    # true = true.numpy()
    # pred = pred.numpy()
    # cm = ConfusionMatrix(y_true=true, y_pred=pred)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # print(type(optimizer).__name__)


