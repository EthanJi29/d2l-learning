import torch

x = torch.arange(4.0, requires_grad=True)
print(x.grad) # 默认是none
y = 2 * torch.dot(x,x)
y.backward()
print(x.grad)

print(x.grad == 4*x)

x.grad.zero_() # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
# 换一个函数
y = x.sum() 
y.backward()
print(x.grad)