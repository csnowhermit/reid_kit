import scipy.io
import torch
import numpy as np
#import time
import os

#######################################################################
# Evaluate
'''
    验证，传参：query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam
    :param qf torch.Tensor，待识别图片的特征
    :param ql numpy.int32，待识别图片的标签
    :param qc numpy.int32，待识别图片的相机ID
    :param gf torch.Tensor，库图片的特征集
    :param gl numpy.ndarray，库图片的标签集
    :param gc numpy.ndarray，库图片的相机ID集
'''
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)    # 重整特征向量为n行1列的张量
    # print(query.shape)
    score = torch.mm(gf,query)    # 矩阵相乘，torch.mul()，矩阵点位相乘
    # print("1、score:", score.shape, score)   # torch.Size([19732, 1])
    score = score.squeeze(1).cpu()    # 将位置1的数据取消掉（位置下标从0开始算）
    # print("2、score:", score.shape, score)   # torch.Size([19732])
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large，排序，升序排序
    # print("3、", type(index), index.shape, index)
    index = index[::-1]
    # print("4、", type(index), index.shape, index)
    # index = index[0:2000]

    # good index，np.argwhere()，返回满足条件的索引值
    query_index = np.argwhere(gl==ql)    # 找到与query有相同label的gallery
    camera_index = np.argwhere(gc==qc)   # 与query有相同camera的gallery

    # print("query_index:", type(query_index), query_index)
    # print("camera_index", type(camera_index), camera_index)

    # 找在query_index，但不在camera_index中的。（即找不同设备下相同label的检索，摒弃相同设备下相同label的）
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)    # assume_unique=True，假设数组中每个值是唯一的

    # 坏的标签
    junk_index1 = np.argwhere(gl==-1)    # 没找到，即为坏的标签
    junk_index2 = np.intersect1d(query_index, camera_index)    # 相同的人在同一摄像头下，
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

'''
    计算mAP：平均精度均值
'''
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()    # 初始化命中率张量，默认为0
    if good_index.size==0:   # if empty，说明库中没找到要query的人，直接返回-1
        cmc[0] = -1
        return ap,cmc    # 这时，精度为0，图片id为-1

    # remove junk_index，在index，但不在junk_index中的
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)    # good_index中每个元素在index中是否存在，存在则标为True，否则为False
    rows_good = np.argwhere(mask==True)    # 返回mark=True的下标
    # print("rows_good:", type(rows_good), rows_good.shape, rows_good)    # numpy.ndarray
    rows_good = rows_good.flatten()    # 将多维数据降为一维，返回数组的拷贝（修改时不影响原数组）
    # print("rows_good:", type(rows_good), rows_good.shape, rows_good)  # numpy.ndarray
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):    # 库中有ngood个匹配上的人
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    return ap, cmc

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')    # 加载gallery和query的特征矩阵
query_feature = torch.FloatTensor(result['query_f'])    # 待查询图像的特征集
query_cam = result['query_cam'][0]                      # 待查询图像的相机ID
query_label = result['query_label'][0]                  # 待查询图像的标签
gallery_feature = torch.FloatTensor(result['gallery_f'])    # 库图像
gallery_cam = result['gallery_cam'][0]                      # 库图像的相机ID
gallery_label = result['gallery_label'][0]                  # 库图像的标签

multi = os.path.isfile('multi_query.mat')    # 是否是个文件，若文件不存在也返回False。（文件存在，则表示开启多进程查询）

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()        # 待查询图片的特征
gallery_feature = gallery_feature.cuda()    # 库图片特征

print(query_feature.shape)
print("len(gallery_label):", len(gallery_label))
CMC = torch.IntTensor(len(gallery_label)).zero_()     # 创建一个指定维度（一维，大小为len(gallery_label)）和类型（Int，置0）的张量
ap = 0.0

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)    # 对于每一个图片，都这么识别
    if CMC_tmp[0]==-1:    # -1表示没识别出来
        continue
    CMC = CMC + CMC_tmp    # 累计每张图片的 CMC_tmp 和 ap_tmp
    ap += ap_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
print(type(CMC), CMC.shape, CMC)

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
