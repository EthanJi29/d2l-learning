import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    # c1~c4 是每条路径的通道数 经过每条路径，宽和高都没变
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path1 single 1x1 conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # path2 1x1 conv -> 3x3 conv padding 1
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # path3 1x1 conv -> 5x5 conv padding 2
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # path4 maxpool 3x3 padding 1 -> 1x1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # h,w 没变，stride=1
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x))) # maxpooling 后不用激活
        return torch.cat((p1, p2, p3, p4), dim=1) # 通道数维度上拼接 最后block输出的通道数为c1~c4的out通道数和
    
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (114, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128), # 最后输出通道1024
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())


# 其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。(卧槽)

net = nn.Sequential(b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    nn.Linear(1024, 10))

X = torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
