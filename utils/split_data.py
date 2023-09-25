from scipy.io import loadmat
import numpy as np
import torch
import os
import math
from copy import deepcopy
from sklearn.model_selection import train_test_split

def transformGT(args, info):
    dataset_path = os.path.join('./datasets', info.path)
    gt_mat = loadmat(os.path.join(dataset_path, info.gt_file_name))[info.gt_mat_name].astype(np.int64)

    known_classes = args.known_classes
    unknown_classes = args.unknown_classes
    unknown_class_index = len(known_classes) + 1

    gt_transform = np.zeros_like(gt_mat)

    for index, cls in enumerate(unknown_classes):
        gt_transform[np.where(gt_mat == cls)] = unknown_class_index

    for index, cls in enumerate(known_classes):
        gt_transform[np.where(gt_mat == cls)] = index + 1

    return gt_transform

def paddingData(dataset, patch_size):
    distance = patch_size // 2
    C, H, W = dataset.shape

    data = torch.zeros(size=(C, H + 2 * distance, W + 2 * distance), dtype=torch.float)
    data[:, distance:distance+H, distance:distance+W] = dataset
    
    data[:, 0:distance, :] = data[:, distance:distance+distance, :].flip(1)
    data[:, -distance:, :] = data[:, -distance*2:-distance, :].flip(1)
    data[:, :, 0:distance] = data[:, :, distance:distance*2].flip(2)
    data[:, :, -distance:] = data[:, :, -distance*2:-distance].flip(2)
    return data

def splitData(gt, args, info):

    known_train_index = []
    known_test_index = []

    for cls in range(1, len(args.known_classes) + 1):
        x, y = np.where(gt == cls)
        locations = x * gt.shape[1] + y

        sample_num = x.shape[0]
        assert args.train_num < sample_num, 'Error: train_num >= sample_num'
        train_num = args.train_num
        if train_num == 0:
            train_num = math.ceil(sample_num * args.train_rate)

        train_index, test_index = train_test_split(locations, train_size=train_num,  random_state=args.seed)
        known_train_index.extend(train_index)
        known_test_index.extend(test_index)

        print('class %d: total %d, train: %d, test: %d' % (cls, sample_num, train_num, sample_num - train_num))

    unknown_test_index = deepcopy(known_test_index)
    unknown_unknown_index = []

    x, y = np.where(gt == len(args.known_classes) + 1)
    locations = x * gt.shape[1] + y
    unknown_test_index.extend(locations)
    unknown_unknown_index.extend(locations)

    print('train: %d, close_test: %d, open_test: %d' % (len(known_train_index), len(known_test_index), len(unknown_test_index)))

    return {
        'known_train_index': known_train_index,
        'known_test_index': known_test_index,
        'unknown_test_index': unknown_test_index,
        'unknown_unknown_index': unknown_unknown_index
    }

def initData(args, info):
    assert args.patch % 2, 'patch shape error'

    data_path = os.path.join('./datasets', info.path)
    data = loadmat(os.path.join(data_path, info.file_name))[info.mat_name].astype(np.float32)
    data = torch.from_numpy(data).permute(2,0,1)
    data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    data = paddingData(data, args.patch)

    return data

def initDataset(args, info):
    
    data = initData(args, info)
    gt = transformGT(args, info)

    dataset_index: dict = splitData(gt, args, info)

    return {
        'data': data,
        'gt': torch.from_numpy(gt),
        **dataset_index
    }