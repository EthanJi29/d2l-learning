import random
import torch
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_example):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:',features[0],'\nlabels:',labels[0])

plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(),1,c='red')
plt.scatter(features[:,(0)].detach().numpy(), labels.detach().numpy(),1,c='green')
plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # random
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    return torch.matmul(X,w)+b

def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2/2 # why /2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad
            param.grad.zero_()

lr = 0.03
num_epochs=3
net=linreg
loss=squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X,w,b),y)
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b), labels)
        print(f'epoch {epoch+1}, loss{float(train_l.mean()):f}')


# 本节只使用张量进行数据存储&线性代数，通过自动微分计算梯度
# 而深度学习库已经提供了很多组件，实现数据迭代器、损失函数、优化器、神经网络层
