import torch
from torch.utils.data import Dataset, DataLoader
from occNerfDataset import OccNerfDataset

import pickle
import os
import json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def createData():
    val_dir = './logs/nusc-sem/val-604/vis_feature/combined.pkl'
    train_dir = './logs/nusc-sem/train-605/vis_feature/combined.pkl'
    dataset_path = './data/nuscenes/nuscenes'

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

    for data in train_data:
        token = "".join(data['index'])
        next_token = pair[token]
        if next_token == "":
            continue
        present_frame.append(data['feature'].cpu())
        # print(token, next_token)
        next_frame.append(train_data[index[next_token]]['feature'].cpu())

    present_train = torch.stack(present_frame, dim=0)
    next_train = torch.stack(next_frame, dim=0)

    present_frame = []
    next_frame = []

    for data in val_data:
        token = "".join(data['index'])
        next_token = pair[token]
        if next_token == "":
            continue
        present_frame.append(data['feature'].cpu())
        # print(token, next_token)
        next_frame.append(val_data[index[next_token]]['feature'].cpu())

    present_val = torch.stack(present_frame, dim=0)
    next_val = torch.stack(next_frame, dim=0)

    # TODO: Add the groundtruth here.
    # Notice: the groundtruth shape should be [num_sample, 6, 900, 1600, 2]

    train_set = OccNerfDataset(present_train.to(device),
                               next_train.to(device),
                            torch.randn(1, 2, 300, 300))
     
    test_set = OccNerfDataset(present_val.to(device),
                              next_val.to(device),
                            torch.randn(1, 2, 300, 300))

    return train_set, test_set
