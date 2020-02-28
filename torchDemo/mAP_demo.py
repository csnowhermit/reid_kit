import numpy as np
import torch

'''
    计算mAP
    calc_mAP()和evaluate_gpu.compute_mAP()方法都行，但同一套算法里一定要统一
'''

index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
good_index = np.array([0, 1, 4, 6])
junk_index = np.array([])

'''
    计算mAP：计算方法不同于evaluate_gpu.compute_mAP
    计算方法：
        检索行人在gallery中有4张图片，在检索得到的list中位置分别在1、2、5、7，则ap为（1/1+2/2+3/5+4/7）/4=0.793；
'''
def calc_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()  # 初始化命中率张量，数值为0
    if good_index.size == 0:  # if empty，说明库中没找到要query的人，直接返回-1
        cmc[0] = -1

    # remove junk_index，在index，但不在junk_index中的
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)  # good_index中每个元素在index中是否存在，存在则标为True，否则为False
    rows_good = np.argwhere(mask == True)  # 返回mark=True的下标
    # print("rows_good:", type(rows_good), rows_good.shape, rows_good)    # numpy.ndarray
    rows_good = rows_good.flatten()  # 将多维数据降为一维，返回数组的拷贝（修改时不影响原数组）
    print("rows_good:", type(rows_good), rows_good.shape, rows_good)  # numpy.ndarray

    cmc[rows_good[0]:] = 1
    for i in range(ngood):  # 库中有5个匹配上的人
        # cmc[rows_good[i]] = 1
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        # if rows_good[i] != 0:
        #     old_precision = i * 1.0 / rows_good[i]
        # else:
        #     old_precision = 1.0
        old_precision = 0.0
        ap = ap + d_recall * (old_precision + precision)  # / 2
        print("第 %d 人，d_recall: %f, rows_good[i]: %f, precision: %f, old_precision: %f, ap: %f" % (i, d_recall, rows_good[i], precision, old_precision, ap))
    return ap, cmc

ap, cmc = calc_mAP(index, good_index, junk_index)
print(ap)
print(type(cmc), cmc.shape, cmc)