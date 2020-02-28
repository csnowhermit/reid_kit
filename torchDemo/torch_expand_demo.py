import torch

'''
    torch.expand(m, n)，将张量扩展为m行n列，每行各列与第一列相同
    m为原张量的列数
'''

x = torch.Tensor([[1], [2], [3]])
print("x:", type(x), x.shape, x.size(), x)   # <class 'torch.Tensor'> torch.Size([3, 1]) torch.Size([3, 1])

y = x.expand(3, 4)
print("x:", type(x), x.shape, x.size(), x)   # <class 'torch.Tensor'> torch.Size([3, 1]) torch.Size([3, 1])
print("y:", type(y), y.shape, y.size(), y)   # <class 'torch.Tensor'> torch.Size([3, 4]) torch.Size([3, 4])

y1 = x.expand(3, 4).t()    # 矩阵转置
print("y1:", type(y1), y1.shape, y1.size(), y1)    # <class 'torch.Tensor'> torch.Size([4, 3]) torch.Size([4, 3])
