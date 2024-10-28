import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# CosineAnnealingLR 利用余弦函数的特点，使学习率在训练过程中按照一个周期性变化的
# 余弦曲线来衰减，即学习率从大到小再到大反复变化。通常用于长时间训练任务，
# 能在训练后期有效避免学习率过快下降。
# T_max: 一个周期内的最大 epoch 数。
# eta_min: 最小学习率。

# 使用场景：
# 适用于训练需要长时间进行的大型模型，如 Transformer 模型（BERT, GPT）
# 和计算机视觉任务中的大型 CNN。在图像分类任务中效果显著。
# 推荐程度：非常推荐，使用的很广泛。

# 模拟模型和优化器
model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 初始化CosineAnnealingLR调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 存储学习率
lrs = []

# 进行训练
for epoch in range(80):
    # 模拟训练步骤
    optimizer.step()
    scheduler.step()

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)

    # 打印学习率
    print(f'Epoch {epoch + 1}: Learning Rate = {current_lr}')

# 绘制学习率曲线
plt.plot(range(1, 81), lrs, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule with CosineAnnealingLR')
plt.grid(True)
plt.show()
