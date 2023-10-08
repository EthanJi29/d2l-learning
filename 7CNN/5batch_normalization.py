# 我们从头开始实现一个具有张量的批量规范化层。
import torch
from torch import nn


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is grad enabled来判断当前是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # test
        X_hat = (X - moving_mean) / torch.sqrt(moving_var+eps)  # 加epsilon小参数，防止/0现象
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:  # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:  # 卷积层
            # 保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下用当前均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新平均移动时的均值和方差
        moving_mean =momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放&移位
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层(pytorch框架不用)
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims==2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # 参与求梯度和迭代的拉伸偏移参数，分别初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的mm和mv
        Y, self.moving_mean, self.moving_var=batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

net = nn.Sequential(nn.Conv2d(), BatchNorm(), nn.Sigmoid(),
                    nn.AvgPool2d(),
                    nn.Conv2d(), BatchNorm(), nn.Sigmoid(),
                    nn.AvgPool2d(),
                    nn.Flatten(),
                    nn.Linear(), BatchNorm(), nn.Sigmoid(),
                    nn.Linear(), BatchNorm(), nn.Sigmoid(),
                    nn.Linear())

# 通常高级API变体运行速度快得多，因为它的代码已编译为C++或CUDA，而我们的自定义代码由Python实现。
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))