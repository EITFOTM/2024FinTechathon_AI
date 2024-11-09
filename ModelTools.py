import thop
import numpy as np
import torch
from ModelEfficientNet import *
from ModelCnn import *
import sklearn.metrics as metrics
# import torch_directml


def update_progress_bar(start_time, loader_len, batch_num, end_time):
    # 更新进度条
    progress_bar_length = 25  # 进度条长度
    batch_time = end_time - start_time
    progress = batch_num / loader_len
    progress_bar = ('>' * int(progress * progress_bar_length) +
                    '-' * (progress_bar_length - int(progress * progress_bar_length)))
    # 在同一行显示进度条
    print(
        f'\r\tBatch {batch_num}/{loader_len} [{progress_bar}], Time: {batch_time:.4f} seconds',
        end="")


def model_load(model_name, device):
    model = globals()[model_name]()  # 找到对应模型并调用它
    model.to(device)
    model_path = f'Model{model_name}.pt'
    model.load_state_dict(torch.load(model_path))
    return model


def model_save(model_name, model):
    model_path = f'Model{model_name}.pt'
    torch.save(model.state_dict(), model_path)


def model_create(model_name, device):
    model = globals()[model_name]()
    model.to(device)
    return model


def get_device(device):
    if device in ['cuda', 'Cuda', 'CUDA']:
        return torch.device('cuda')
    # elif device in ['gpu', 'GPU', 'Gpu']:
    #     return torch_directml.device(0)
    else:
        return torch.device('cpu')


def evaluation(model, device):
    """
    深度学习模型参数量/计算量和推理速度计算
    https://zhuanlan.zhihu.com/p/376925457?utm_id=0
    Evaluate model
    :param model:
    :param device:
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
    model1 = model_load('efficientnet_b0', device)
    evaluation(model1, device)
    true = torch.tensor([1,1,0,0,1,0,1,0,1])
    pred = torch.tensor([1,1,1,1,1,0,0,0,0])
    print(type(true))
    true = true.numpy()
    pred = pred.numpy()
    cm = ConfusionMatrix(y_true=true, y_pred=pred)

