### Tensor 是什么

就是 PyTorch 里的多维数组。对应你脑子里的 `float array[][]`。可以跑在 GPU 上，支持自动求导。

---

### 创建 Tensor

```python
import torch

# 从列表创建
a = torch.tensor([1, 2, 3])                    # 一维，形状 [3]
b = torch.tensor([[1, 2], [3, 4]])             # 二维，形状 [2, 2]

# 创建特殊 Tensor
c = torch.zeros(3, 4)                          # 全 0，形状 [3, 4]
d = torch.ones(2, 3)                           # 全 1，形状 [2, 3]
e = torch.randn(4, 8)                          # 随机高斯，形状 [4, 8]
f = torch.randint(0, 65, (4, 8))              # 随机整数，范围 [0, 65)，形状 [4, 8]
```

---

### 查看形状

```python
print(e.shape)   # torch.Size([4, 8])
print(e.size())  # 同上
print(e.dtype)   # torch.float32
```

---

### 索引和切片（和 C 数组一样）

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])   # [2, 3]

a[0, 1]         # 第 0 行第 1 列，值 = 2
a[0]            # 第 0 行，形状 [3]
a[:, 1]         # 所有行的第 1 列，形状 [2]
a[0, 0:2]       # 第 0 行，前两列，形状 [2]
```

---

### 形状变换

```python
a = torch.randn(4, 8)        # [4, 8]
b = a.view(32)               # 展平成一维，[32]
c = a.view(2, 16)            # 改成 [2, 16]
d = a.view(-1, 4)            # -1 让 PyTorch 自动算，这里变成 [8, 4]
e = a.T                      # 转置，[8, 4]
```

---

### 运算

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

c = a + b                    # 逐元素加
d = a * b                    # 逐元素乘
e = a @ b.T                  # 矩阵乘法，[3, 4] @ [4, 3] = [3, 3]
f = a.sum()                  # 所有元素求和，返回标量
g = a.sum(dim=1)             # 按行求和，形状 [3]
```

---

### 数学函数

```python
a = torch.randn(3, 4)

b = torch.exp(a)             # e^a，逐元素
c = torch.log(a)             # ln(a)，逐元素
d = torch.softmax(a, dim=1)  # Softmax，沿指定维度
e = a / torch.sqrt(tensor)   # 除以标量，对应 sqrt(d_k)
```

---

### 类型转换

```python
a = torch.tensor([1.5, 2.7])      # float32
b = a.long()                       # 转 int64
c = a.float()                      # 转 float32
d = a.int()                        # 转 int32
```

---

### 搬到 GPU

```python
device = torch.device('cuda')
a = a.to(device)                   # 搬到 GPU
b = torch.randn(3, 4).to(device)   # 直接在 GPU 上创建
```

---

### 和 NumPy 互转

```python
a = torch.tensor([1, 2, 3])
b = a.numpy()                      # Tensor -> NumPy

import numpy as np
c = np.array([1, 2, 3])
d = torch.from_numpy(c)            # NumPy -> Tensor
```
