import torch
x=torch.rand(5,3)
y=torch.rand(5,3)
# print(x,y)
# 1.tensor.view
z=x.view(-1,15)
# print(z)
# tensor.item
z=z[0,2].item()
# print(z)
# 使用CUDA
if torch.cuda.is_available():
    device=torch.device("cuda") #一个cuda设备对象
    y=torch.ones_like(x,device=device)#直接从GPU创建张量
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))#.to也能对类型做更改