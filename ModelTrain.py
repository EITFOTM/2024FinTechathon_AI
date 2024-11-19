import time
from ModelTools import *
import torch.optim as optim
import torch.backends.cudnn
# -i https://pypi.tuna.tsinghua.edu.cn/simple


def train(d: str = 'cpu',
          model_name: str = 'Cnn',
          optimizer_name: str = 'SGD',
          learning_rate: float = 0.01,
          n_epochs: int = 1,
          pre_epochs: int = None) -> None:
    """

    :param d: 在该设备上进行模型的训练
    :param model_name: 选择训练时所使用的模型
    :param optimizer_name: 选择的优化器名字
    :param learning_rate: 设置学习率
    :param n_epochs: 本次训练所进行的轮次数
    :param pre_epoch: 是否需要加载已经训练过的模型继续训练，如果是，则需要填上次训练累计的训练轮次数
    """
    #
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 加载训练集
    print('Loading data...')
    root_dir = "data"
    datasets_dir = ["Face2", "Face3"]
    data_type = ["Train", "Valid"]
    classes = ["Fake", "Real"]
    batch = 16
    # 图像增强详见：https://blog.csdn.net/weixin_46334272/article/details/135395701
    # 进行数据预处理和增强操作:通过对训练集进行各种变换和扩增操作，可以增加训练数据的多样性和丰富性，从而提高模型的泛化能力。
    # 数据增强的目的是通过对训练集中的图像进行随机变换，生成更多样的图像样本，以模拟真实世界中的各种场景和变化。
    # 这样可以帮助模型学习到更多不同的特征和模式，提高模型对于输入数据的扰动、噪声、异常值等干扰的抵抗能力，同时减少过拟合的风险。
    # 常见的数据增强操作包括图像翻转、旋转、缩放、裁剪、平移、变换色彩空间、添加噪声等。增加样本的多样性，可以更好地训练出泛化能力强的模型
    data_transforms = {
        'Train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45度到45度之间
                                     transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转，10%的概率
                                     transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转，10%的概率
                                     # 参数分别为亮度，对比度，饱和度，色相，详细可参考：https://blog.csdn.net/lxhRichard/article/details/128083192
                                     transforms.ColorJitter(0.2, 0.1, 0.1, 0.1),
                                     transforms.RandomGrayscale(p=0.01),  # 随机转换成灰度率， 3通道就是R=G=B
                                     transforms.ToTensor()
                                     ]),
        'Valid': transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()
                                     ])
    }
    loaders, dataset_size = datasets_load(root_dir, datasets_dir, data_type, classes, batch, data_transforms)

    # 创建模型
    device = get_device(d)
    print(f'You are training on the {device}.')
    if pre_epochs is None:
        model = model_create(model_name, device)
    else:
        model, best_acc, optimizer_state_dict, history = model_load(model_name, device, d, optimizer_name, pre_epochs)

    # 优化器详解可参考：https://blog.csdn.net/2301_76846375/article/details/141476689
    if optimizer_name == 'Adam':
        # Adam 优化器是一种自适应学习率的优化算法，结合了动量梯度下降和 RMSprop 算法的思想。
        # 它通过自适应地调整每个参数的学习率，从而在训练过程中加速收敛。
        # 所以Adam优化器的初始学习率可以随意设置，也不需要再添加周期变化的学习率
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        # 随机梯度下降（SGD）是一种梯度下降形式，对于每次前向传递，都会从总的数据集中随机选择一批数据，即批次大小1。
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_name = optimizer.__class__.__name__
    if pre_epochs is None:
        best_acc = 0  # 记录最好的正确率
        train_acc_history = []
        valid_acc_history = []
        train_loss_history = []
        valid_loss_history = []
    else:
        optimizer.load_state_dict(optimizer_state_dict)
        best_acc = best_acc.item()
        train_acc_history = history['train_acc_history']
        valid_acc_history = history['valid_acc_history']
        train_loss_history = history['train_loss_history']
        valid_loss_history = history['valid_loss_history']

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 基于余弦函数周期变化的学习率
    # 如果最后一层有LogSoftmax()，则不能用nn.CrossEntropyLoss()，因为nn.CrossEntropyLoss()相当于LogSoftmax()和nn.NLLLoss()整合
    if model_name == 'Cnn':
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    lrs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())  # 保存效果最好的模型
    # 开始训练模型
    start_time = time.time()
    for epoch in range(n_epochs):
        if pre_epochs is None:
            print(f'Epoch {epoch + 1}/{n_epochs}')
        else:
            print(f'Epoch {epoch + 1 + pre_epochs}/{n_epochs + pre_epochs}')
        print('-' * 10)
        for phase in data_type:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            phase_start_time = time.time()
            for i, (images, labels) in enumerate(loaders[phase], 0):
                batch_start_time = time.time()
                loader_len = len(loaders[phase])
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 清零
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(images)
                    result_loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # 训练时更新权重
                    if phase == 'Train':
                        result_loss.backward()
                        optimizer.step()
                        scheduler.step()
                # 计算损失
                current_loss = result_loss.item() * images.size(0)
                running_loss += current_loss
                running_corrects += torch.sum(preds == labels.data)
                batch_end_time = time.time()

                # 更新进度条
                batch_num = i + 1
                update_progress_bar(phase, batch_start_time, loader_len, batch_num, batch_end_time, current_loss)

            phase_loss = running_loss / dataset_size[phase]
            phase_acc = running_corrects.double() / dataset_size[phase]
            phase_end_time = time.time()
            epoch_time = phase_end_time - phase_start_time
            print(f'\n{phase} mean loss: {phase_loss:.4f}, total loss: {running_loss:.0f}, acc: {phase_acc:.4f}.')
            epoch_minutes, epoch_seconds = divmod(epoch_time, 60)
            # 和epoch+1和gup信息
            print(f'Total time spent on this epoch: {epoch_minutes:.0f} minutes {epoch_seconds:.1f} seconds')
            # 得到最好那次的模型
            if phase == 'Valid':
                valid_acc_history.append(phase_acc)
                valid_loss_history.append(phase_loss)
            if phase == 'Train':
                train_acc_history.append(phase_acc)
                train_loss_history.append(phase_loss)
            if phase == 'Valid' and phase_acc > best_acc:
                best_acc = phase_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'train_acc_history': train_acc_history,
                         'valid_acc_history': valid_acc_history,
                         'train_loss_history': train_loss_history,
                         'valid_loss_history': valid_loss_history
                         }
                model_save(model_name, state, optimizer_name, epoch+1)
        print('Optimizer learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        lrs.append(optimizer.param_groups[0]['lr'])
        print()

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f'Total time: {minutes:.0f} minutes {seconds:.1f} seconds.')
    print(f'Best accuracy: {best_acc:.4f}')

    # 保存模型最好的那一次模型
    model.load_state_dict(best_model_wts)
    state = {'state_dict': model.state_dict(),
             'best_acc': best_acc,
             'optimizer': optimizer.state_dict(),
             'train_acc_history': train_acc_history,
             'valid_acc_history': valid_acc_history,
             'train_loss_history': train_loss_history,
             'valid_loss_history': valid_loss_history
             }
    # 模型保存时增加优化器名字，轮数epoch
    if pre_epochs is None:
        model_save(model_name, state, optimizer_name, n_epochs)
    else:
        model_save(model_name, state, optimizer_name, n_epochs + pre_epochs)


if __name__ == "__main__":
    # model_name = 'efficientnet_b0'
    model_name = 'Vgg16'  # Vgg16模型不能用Adam优化器，否则可能会出现损失异常大的情况
    optimizer_name = "SGD"
    lr = 0.01
    epochs = 2
    pre_epochs = None
    d = 'cpu'
    # NVIDIA显卡用"cuda"，没有显卡用"cpu"
    train(d=d, model_name=model_name, optimizer_name=optimizer_name,
          learning_rate=lr, n_epochs=epochs, pre_epochs=pre_epochs)
    print("=" * 150)
