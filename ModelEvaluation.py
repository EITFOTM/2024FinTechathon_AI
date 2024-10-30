import thop
import torch
import numpy as np


def evaluation(model, device):
    """
    深度学习模型参数量/计算量和推理速度计算
    https://zhuanlan.zhihu.com/p/376925457?utm_id=0
    Evaluate model
    :param model:
    :param device:
    """
    # FLOPs和Params计算
    optimal_batch_size = 1
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
    print(mean_syn)

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

