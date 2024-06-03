import torch
from torch.utils.data import Dataset, DataLoader

class OccNerfDataset(Dataset):
    def __init__(self, input1_data, input2_data, label):
        self.input1_data = input1_data # 1x18x300x300x24
        self.input2_data = input2_data # 1x18x300x300x24
        self.label = label #1x2x300x300

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        input1 = self.input1_data[idx]
        input2 = self.input2_data[idx]
        label = self.label[idx]
        return input1, input2, label
