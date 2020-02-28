import torch

'''
    torch.Tensor.view()，重整Tensor，维度变换，拿不准行数或列数的话用-1代替
    x.view(-1, 1)，重整Tensor为n行1列的张量
    x.view(1, -1)，重整Tensor为1行n列的张量
'''

x = torch.randn(4, 5)    # 随机生成4行5列的张量

print(type(x))    # torch.Tensor
print('tensor原型:',x)

# print('tensor维度变换，由（4,5）到（20,1）:',x.view(20, 1))
# #由（4,5）到（-1,1）的tensor维度变换，其中-1是tensor在1下的另一个维度的大小，即为20/1=20，也就是说在这里-1=20
# print('tensor维度变换，由（4,5）到（-1,1）:',x.view(-1, 1))

print('tensor维度变换，由（4,5）到（1,20）:',x.view(1, 20))
#由（4,5）到（1, -1）的tensor维度变换，其中-1是tensor在1下的另一个维度的大小，即为20/1=20，也就是说在这里-1=20
print('tensor维度变换，由（4,5）到（1,-1）:',x.view(1, -1))
