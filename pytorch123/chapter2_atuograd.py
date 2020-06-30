import torch
# 使用.requires_grad来追踪Tensor的操作
x=torch.ones(2,2,requires_grad=True)
# print(x)
y=x+2
# print(y)
y=x+2
# print(y)
y=y**2
# print(y.grad_fn)
# 梯度
x=torch.tensor(([[2.0,4.0],[3.0,5.0]]),requires_grad=True)
y=3*(x+2)**2
l=y.mean()
print(l)
l.backward()
print(x.grad)