# 从入门到放弃pytorch

https://github.com/zergtant/pytorch-handbook

# 1.Tensor概念

​	**任何以‘’_‘'结尾的的操作对会用结果替换原变量，如x.copy_(y)**

### 1.1 tensor.view

​		tensor.view返回一个数据相同，但形状不同的tensor

```python
x=torch.rand(5,3)
y=x.view(-1,15)
-->tensor([[0.3007, 0.6886, 0.7683, 0.3512, 0.9341, 0.6860, 0.4841, 0.4691, 0.4791, 0.4585, 0.8181, 0.5360, 0.2125, 0.1982, 0.0844]])
```

### 1.2 tensor.item

​		返回只有一个元素tensor的值

```python
#接上文
y=y[0,1].item
-->0.6886
```



### 1.3 CUDA张量

​			使用`.to`方法可以将Tensor移动到任何设备中

```python
if torch.cuda.is_available():
    device=torch.device("cuda") #一个cuda设备对象
    y=torch.ones_like(x,device=device)#直接从GPU创建张量
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))#.to也能对类型做更改
-->tensor([[1.3507]], device='cuda:0')
-->tensor([[1.3507]], dtype=torch.float64)
```





## 2. 自动求导机制*

​		**autograd包是PtTorch中所有神经网络的核心**。该包为在Tensor上的所有操作提供自动求导。*它是一个在**运行时定义**的框架，意味着反向传播是根据的你的代码来确定如何运行，每次迭代可以是不同的*。理解自动求导机制可以帮助我们写出更高效、更干净的程序，且能帮助我们调试。

### 2.1 张量（Tensor）

​		`torch.Tensor`是这个包的核心类，**如果设置`.requires_grad`为`True`，那么将会追踪所有对该Tensor的操作。当完成计算后通过调用`.backward()`自动计算所有的梯度，**这个张量的所有梯度将会自动积累到`.grad`的属性。

```python
import torch
# 使用.requires_grad来追踪Tensor的操作
x=torch.ones(2,2,requires_grad=True)
y=x+2
print(y)
-->tensor([[3., 3.],[3., 3.]], grad_fn=<AddBackward0>)
y=y*y
print(y)
-->tensor([[9., 9.],[9., 9.]], grad_fn=<MulBackward0>)
x=x**2
print(x)
-->tensor([[1., 1.],[1., 1.]], grad_fn=<PowBackward0>)
```

​		要阻止张量追踪历史（和使用内存）可以将代码块包装在`with torch.no_grad：`中。**这在评估模型中特别有用，因为评估时我们不需要计算梯度。**

​		autograd根据用户的对Tensor的操作构建计算图，对Tensor的操作抽象为`Function`。对于那些不是任何函数（Function）的输出，由用户创建的Tensor，这个Tensor的`grad_fn`是`None`。

​		在向量微积分中，`雅各比矩阵`是`一阶偏导数`以一定的方式排列成的矩阵，设
$$
y=\frac{3}{4}(x+2)^2\\
\frac{\partial{y}}{\part{x}}=\frac{3}{2}(x+2)\\
if\ x,y \ is \ matrix,then\\
\frac{\partial{y_i}}{\part{x_i}}=\frac{3}{2}(x_i+2)
$$
将$\frac{\part{y_i}}{\part{x_i}}$写成矩阵就得到一阶偏导数的雅克比矩阵：

![jacob](src\jacob.svg)



**一般来说,`torch.autograd`**就是用来计算`vector-Jacobian product`的工具。

**给定$\vec{v}=\{v_1,v_2,...,v_m\}$，如果$\vec{v}$恰好是函数$l=g(y)$的梯度，即$\vec{v}=\{\frac{\part{l}}{\part{y_1}},\frac{\part{l}}{\part{y_2}},...\frac{\part{l}}{\part{y_m}}\}$,根据链式法则($\frac{\part{y}}{\part{x}}\frac{\part{l}}{\part{y}}=\frac{\part{l}}{\part{x}}$)：**

$$
J^{T}\cdot v = \begin{pmatrix} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{pmatrix} \begin{pmatrix} \frac{\partial l}{\partial y_{1}}\\ \vdots \\ \frac{\partial l}{\partial y_{m}} \end{pmatrix} = \begin{pmatrix} \frac{\partial l}{\partial x_{1}}\\ \vdots \\ \frac{\partial l}{\partial x_{n}} \end{pmatrix}
$$

则$J^t·v$是$\vec{x}$对$l$的梯度。

```python
x=torch.tensor(([[2.0,4.0],[3.0,5.0]]),requires_grad=True)
y=3*(x+2)**2
l=y.mean()#l=(1/4)*y
print(l)
l.backward()
print(x.grad)
-->tensor(94.5000, grad_fn=<MeanBackward0>)
-->tensor([[ 6.0000,  9.0000],
        [ 7.5000, 10.5000]])
```

**使用`.backward()`的时候，确保是标量scalar。求导只能是*标量*对*标量*，或者*标量*对*向量/矩阵*求导**。否则会报如下错误：

```python
RuntimeError: grad can be implicitly created only for scalar outputs
```

