import pickle
import os
import json
import torch

val_dir = './logs/nusc-sem/val-604/vis_feature/combined.pkl'
train_dir = './logs/nusc-sem/train-605/vis_feature/combined.pkl'
dataset_path = './data/nuscenes/nuscenes'


from nuscenes.nuscenes import NuScenes

# 初始化 NuScenes 对象
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes/nuscenes', verbose=True)

# 给定的索引
index_temporal = '12e1aa58e9d04068a33f37b896b901a3'
rec = nusc.get('sample', index_temporal)
print(rec['data']['CAM_BACK'])

# 获取样本中的数据（如图像）
sample_data = nusc.get('sample_data', rec['data']['CAM_BACK'])

# 获取图像文件名
image_filename = sample_data['filename']
print(image_filename)
# pair = {}
#
# mini_train = \
#     ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
#
# mini_val = \
#     ['scene-0103', 'scene-0916']
#
# with open(os.path.join(dataset_path, 'v1.0-mini', 'sample.json'), 'r') as f:
#     samples = json.load(f)
#
# with open(os.path.join(dataset_path, 'v1.0-mini', 'scene.json'), 'r') as f:
#     scenes = json.load(f)
#
# scene_map = {scene['token']: scene['name'] for scene in scenes}

# print(len(samples))

# train = []
# val = []
#
# for sample in samples:
#     scene_name = scene_map.get(sample['scene_token'], 'Unknown')
#     print(scene_name)
#     if scene_name in mini_train:
#         train.append(sample['token'])
#     elif scene_name in mini_val:
#         val.append(sample['token'])
#     else:
#         print(sample['token'], "somethings went wrong !")
#         break
#
# with open("./datasets/nusc/mini_train.txt", 'w') as f:
# # with open("./mini_train.txt", 'w') as f:
#     for token in train:
#         f.write(token+'\n')

# for sample in samples:
#     pair[sample['token']] = sample['next']
# #
# with open(train_dir, 'rb') as f:
#     train_data = pickle.load(f)
#
# index = {}
#
# for idx, data in enumerate(train_data):
#     token = "".join(data['index'])
#     index[token] = idx
#
# # with open(val_dir, 'rb') as f:
# #     val_data = pickle.load(f)
#
# present = []
# next = []
#
# for data in train_data:
#     token = "".join(data['index'])
#     next_token = pair[token]
#     if next_token == "":
#         continue
#     present.append(data['feature'].cpu())
#     # print(token, next_token)
#     next.append(train_data[index[next_token]]['feature'].cpu())
#
# present = torch.stack(present)
# print(present.shape)
# print(present[0].shape)

# import torch
# from torch.utils.data import Dataset, DataLoader
# from occNerfDataset import OccNerfDataset


# def createData():
#     train_set = OccNerfDataset(torch.randn(1, 64, 84, 168),
#                                torch.randn(1, 64, 84, 168),
#                                torch.randn(1, 2, 300, 300))
#
#     test_set = OccNerfDataset(torch.randn(1, 64, 84, 168),
#                               torch.randn(1, 64, 84, 168),
#                               torch.randn(1, 2, 300, 300))
#
#     return train_set, test_set
#
# class OccNerfDataset(Dataset):
#     def __init__(self, input1_data, input2_data, label):
#         self.input1_data = input1_data # 1x18x300x300x24
#         self.input2_data = input2_data # 1x18x300x300x24
#         self.label = label #1x2x300x300
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         input1 = self.input1_data[idx]
#         input2 = self.input2_data[idx]
#         label = self.label[idx]
#         return input1, input2, label
