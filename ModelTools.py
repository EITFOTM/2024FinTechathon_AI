import thop
import numpy as np
import torch
from ModelEfficientNet import *
from ModelCnn import *


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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = model_load('efficientnet_b0', device)
    evaluation(model1, device)
