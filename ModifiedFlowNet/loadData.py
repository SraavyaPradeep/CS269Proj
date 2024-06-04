import torch
from torch.utils.data import Dataset, DataLoader
from occNerfDataset import OccNerfDataset

def createData():
    train_set = OccNerfDataset(torch.randn(1, 64, 84, 168), 
                            torch.randn(1, 64, 84, 168), 
                            torch.randn(1, 2, 300, 300))
     
    test_set = OccNerfDataset(torch.randn(1, 64, 84, 168), 
                            torch.randn(1, 64, 84, 168), 
                            torch.randn(1, 2, 300, 300))

    return train_set, test_set
