import torch
from torch import nn, optim
import torchvision
import sys
import time

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# ratio = 4
# small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
lr = 0.05
num_epochs = 10
batch_size = 128
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # convolutional layers part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         nn.Linear(out_channels*7*7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10))


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        # mac跑不了4
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=False, transform=transform)

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 没指定device 就用net对device
        device = list(net.parameters())[0].device
    acc_sum, n = 0., 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式，关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) ==
                        y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum/n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    for eopch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0., 0., 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().sum()
            train_acc_sum += (y_hat.argmax(dim=1) ==
                              y).sum().cpu().item()  # 预测结果第一个和label比较
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (eopch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))


net = vgg(conv_arch)
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
