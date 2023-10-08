import math
import numpy as np
import torch
from torch import nn 
from d2l import torch as d2l
import matplotlib.pyplot as plt

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1,-1)) # 
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i+1) # gamma函数 gamma(n)=(n-1)!
# labels维度（n_train+n_test, )
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)  

# numpy ndarray转换为tensor 
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
# print(features[:2], poly_features[:2, :], labels[:2])
his_train_loss = []
his_test_loss = []
his=[]

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0]/metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none') # 返回张量
    input_shape = train_features.shape[-1] # why -1
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr = 0.01)
    
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        
        if epoch==0 or (epoch+1)%20==0:
            print('train_loss=', evaluate_loss(net, train_iter, loss), 'test_loss=', evaluate_loss(net, test_iter, loss), '\n')
            his_train_loss.append(evaluate_loss(net, train_iter, loss))
            his_test_loss.append(evaluate_loss(net, test_iter, loss))
            his.append(epoch)
        #print('weight:', net[0].weight.data.numpy())

# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:]) # 学的参数太多，过拟合

plt.plot(his, his_train_loss, scaley='log', label='train_loss', c='red')
plt.plot(his, his_test_loss, scaley='log', label='test_loss', c='green')
plt.yscale('log')
plt.legend()
plt.show()    
# print(his_train_loss)