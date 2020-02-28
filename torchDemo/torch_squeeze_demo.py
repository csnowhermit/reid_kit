import torch
import matplotlib.pyplot as plt

'''
    torch.squeeze(Tensor, dim=n)：减少数据维度，直接将n位置的维度数据去掉，其他往前移
    torch.unsqueeze(Tensor, dim=n)：扩充数据维度，在n起的指定位置增加一个维度，后续向后串
'''

a = torch.randn(2, 3)  # 标准正态分布生成随机数，2行3列
print(a)
print(a.shape)  # torch.Size([2, 3])

# unsqueeze:扩充数据维度，在0起的指定位置N加上维数为一的维度
b = torch.unsqueeze(a, 1)  # [2, 3]中在位置1，就是=3的位置增加维度1，3向后串
print(b.shape)  # torch.Size([2, 1, 3])
print(b)

c = torch.unsqueeze(a, 0)  # [2, 3]中在位置0，就是=1的位置增加维度1，2向后串
print(c.shape)  # torch.Size([1, 2, 3])
print(c)
# --------------------------------------------------------------#
f = torch.randn(3)
print(f)
print(f.shape)  # torch.Size([3])
g = f.unsqueeze(0)  # [3]中在位置0，就是=3的位置增加维度1，3向后串
print(g.shape)  # torch.Size([1, 3])
# --------------------------------------------------------------#
# squeeze:维度压缩，在0起的指定位置，去掉维数为1的的维度
d = torch.squeeze(c, dim=0)  # d=c.squeeze(0)
print(d)
print(d.shape)  # torch.Size([2, 3])
