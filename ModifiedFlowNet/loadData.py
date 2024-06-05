import torch
from torch.utils.data import Dataset, DataLoader
from occNerfDataset import OccNerfDataset

import pickle
import os
import json
import numpy as np

from nuscenes.nuscenes import NuScenes

# 初始化 NuScenes 对象
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes/nuscenes', verbose=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cam_list = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def createData():
    val_dir = './logs/nusc-sem/val-604/vis_feature/combined.pkl'
    train_dir = './logs/nusc-sem/train-605/vis_feature/combined.pkl'
    gt_dir = './data/nuscenes/nuscenes/'
    dataset_path = './data/nuscenes/nuscenes-optical-flow'

    pair = {}

    with open(os.path.join(dataset_path, 'v1.0-mini', 'sample.json'), 'r') as f:
        samples = json.load(f)

    for sample in samples:
        pair[sample['token']] = sample['next']

    with open(train_dir, 'rb') as f:
        train_data = pickle.load(f)

    with open(val_dir, 'rb') as f:
        val_data = pickle.load(f)

    index = {}
    for idx, data in enumerate(train_data):
        token = "".join(data['index'])
        index[token] = idx

    present_frame = []
    next_frame = []
    gt_train = []

    for data in train_data:
        token = "".join(data['index'])
        next_token = pair[token]
        if next_token == "":
            continue
        present_frame.append(data['feature'].cpu())
        # print(token, next_token)
        next_frame.append(train_data[index[next_token]]['feature'].cpu())

        rec = nusc.get('sample', token)
        gt = []
        for idx, cam in cam_list:
            cam_upper = ('cam_' + cam).upper()
            sample_data = nusc.get('sample_data', rec['data'][cam_upper])
            image_filename = sample_data['filename'].split('/')[-1]
            path = os.path.join(gt_dir, cam, image_filename)
            gt.append(torch.tensor(readFlow(path)))
        gt = torch.stack(gt, dim=0)
        gt_train.append(gt)

    present_train = torch.stack(present_frame, dim=0)
    next_train = torch.stack(next_frame, dim=0)
    gt_train = torch.stack(gt_train, dim=0)

    present_frame = []
    next_frame = []
    gt_val = []

    for data in val_data:
        token = "".join(data['index'])
        next_token = pair[token]
        if next_token == "":
            continue
        present_frame.append(data['feature'].cpu())
        # print(token, next_token)
        next_frame.append(val_data[index[next_token]]['feature'].cpu())

        rec = nusc.get('sample', token)
        gt = []
        for idx, cam in cam_list:
            cam_upper = ('cam_' + cam).upper()
            sample_data = nusc.get('sample_data', rec['data'][cam_upper])
            image_filename = sample_data['filename'].split('/')[-1]
            path = os.path.join(gt_dir, cam, image_filename)
            gt.append(torch.tensor(readFlow(path)))
        gt = torch.stack(gt, dim=0)
        gt_val.append(gt)

    present_val = torch.stack(present_frame, dim=0)
    next_val = torch.stack(next_frame, dim=0)
    gt_val = torch.stack(gt_val, dim=0)

    # TODO: Add the groundtruth here.
    # Notice: the groundtruth shape should be [num_sample, 6, 900, 1600, 2]

    train_set = OccNerfDataset(present_train.to(device),
                               next_train.to(device),
                            gt_train.to(device))
     
    test_set = OccNerfDataset(present_val.to(device),
                              next_val.to(device),
                            gt_val.to(device))

    return train_set, test_set
