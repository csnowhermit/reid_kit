# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test

'''
    测试模型
    准备测试数据，真正验证脚本在evaluate_gpu.py中
'''

if __name__ == '__main__':
    # fp16
    try:
        from apex.fp16_utils import *
    except ImportError:  # will be 3.x series
        print(
            'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    ######################################################################
    # Options
    # --------

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121')
    parser.add_argument('--PCB', action='store_true', help='use PCB')
    parser.add_argument('--multi', action='store_true', help='use multiple query')    # 多进程查询
    parser.add_argument('--fp16', action='store_true', help='use fp16.')
    parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()
    ###load config###
    # load the training config
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    opt.fp16 = config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.stride = config['stride']

    if 'nclasses' in config:  # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 751

    str_ids = opt.gpu_ids.split(',')
    # which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    print("opt:", opt)
    print('We use the scale: %s' % opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ############### Ten Crop
        # transforms.TenCrop(224),
        # transforms.Lambda(lambda crops: torch.stack(
        #   [transforms.ToTensor()(crop)
        #      for crop in crops]
        # )),
        # transforms.Lambda(lambda crops: torch.stack(
        #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        #       for crop in crops]
        # ))
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_dir = test_dir

    if opt.multi:    # 多进程查询，找multi-query目录
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query', 'multi-query']}
        # num_workers=16时报错：ImportError: DLL load failed: 页面文件太小，无法完成操作。
        # 解决方案：num_workers=0
        # torch.utils.data.DataLoader，数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
        # 在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直到把所有的数据都抛出。就是做一个数据的初始化。
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=0) for x in
                       ['gallery', 'query', 'multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=0) for x in ['gallery', 'query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    print("dataloaders:", type(dataloaders), dataloaders)
    print("class_names:", type(class_names), class_names)


    ######################################################################
    # Load model
    # ---------------------------
    def load_network(network):
        save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network


    ######################################################################
    # Extract feature，提取特征
    # ----------------------
    #
    # Extract feature from  a trained model.，从训练好的模型中提取特征
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        count = 0    # 标记已经拿了多少个图片
        for data in dataloaders:
            img, label = data
            print(img.shape)    # torch.Size([opt.batchsize, 3, 256, 128])
            n, c, h, w = img.size()   # opt.batchsize，通道，高，宽
            count += n
            print("current count:", count)
            ff = torch.FloatTensor(n, 512).zero_().cuda()
            if opt.PCB:
                ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts

            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                              align_corners=False)
                    outputs = model(input_img)
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff.data.cpu()), 0)
        return features

    '''
        获取相机ID和标签
        :param img_path 图片路径
        :returns camera_id, labels 相机ID，标签
    '''
    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            # filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels


    gallery_path = image_datasets['gallery'].imgs   # 查询库图像
    query_path = image_datasets['query'].imgs       # 查询图片

    gallery_cam, gallery_label = get_id(gallery_path)    # 查询库图像的每张图片的相机ID，标签
    query_cam, query_label = get_id(query_path)          # 查询图片中每张图片的相机ID，标签

    if opt.multi:    # 指定多进程查询的话，找multi-query目录
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    # 1.先load模型结构
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses)
    else:
        model_structure = ft_net(opt.nclasses, stride=opt.stride)    # 加载模型及诶够

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    # if opt.fp16:
    #    model_structure = network_to_half(model_structure)

    # 2.再load weight
    model = load_network(model_structure)    # 加载权重值weight

    # Remove the final fc layer and classifier layer
    if opt.PCB:
        # if opt.fp16:
        #    model = PCB_test(model[1])
        # else:
        model = PCB_test(model)
    else:
        # if opt.fp16:
        # model[1].model.fc = nn.Sequential()
        # model[1].classifier = nn.Sequential()
        # else:
        model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    with torch.no_grad():    # torch.no_grad()，禁用渐变计算的上下文管理器
        gallery_feature = extract_feature(model, dataloaders['gallery'])    # 库（特征）
        query_feature = extract_feature(model, dataloaders['query'])        # 待查询图片（特征）
        if opt.multi:
            mquery_feature = extract_feature(model, dataloaders['multi-query'])

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(),     # 库的特征集
              'gallery_label': gallery_label,           # 库的标签集
              'gallery_cam': gallery_cam,               # 库的相机ID集
              'query_f': query_feature.numpy(),         # 待查询图片的特征集
              'query_label': query_label,               # 待查询图片的标签集
              'query_cam': query_cam}                   # 待查询图片的相机ID集
    scipy.io.savemat('pytorch_result.mat', result)        # 以上数据保存至文件

    print(opt.name)
    result = './model/%s/result.txt' % opt.name
    print(result)
    # print('python evaluate_gpu.py | tee -a %s' % result)
    # os.system('python evaluate_gpu.py | tee -a %s' % result)    # | tee，重定向到文件（类同于>>重定向），-a表示追加
    # windows下没有tee命令，在此换用>>重定向
    print('python evaluate_gpu.py >> %s' % result)
    os.system('python evaluate_gpu.py >> %s' % result)

    if opt.multi:
        result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
        scipy.io.savemat('multi_query.mat', result)
