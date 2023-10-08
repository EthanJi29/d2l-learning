import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10))

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)

# net.apply(func) 接受一个函数 func 作为参数，然后将这个函数应用到模型 net 的所有子模块（即网络的所有层）。
net.apply(init_weights) 
batch_size = 256
lr=0.1
num_epochs=10
loss = nn.CrossEntropyLoss(reduction='none') # ‘mean’（默认）返回均值，标量； ‘sum’返回和，标量； ‘none’返回张量
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter,loss,num_epochs,trainer)




