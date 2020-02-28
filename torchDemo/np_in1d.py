import numpy as np

'''
    np.in1d(A, B)，以A为基准，当前位置元素在B中出现，则标识True，否则标识False
    参数 invert=True，标识标记完之后倒置（结果True改False，False改True）
'''

A = np.array([10,4,6,7,1,5,3,24,9,10,18])
B = np.array([1,8,9])

C = np.in1d(A, B)
print(C)    # [False False False False  True False False False  True False False]

D = np.in1d(B, A)
print(D)    # [ True False  True]

E = np.in1d(A, B, invert=True)
print(E)    # [ True  True  True  True False  True  True  True False  True  True]