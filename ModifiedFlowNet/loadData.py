import torch
from torch.utils.data import Dataset, DataLoader
from occNerfDataset import OccNerfDataset

def createData():
    train_set = OccNerfDataset(torch.randn(1, 18, 300, 300, 24), 
                            torch.randn(1, 18, 300, 300, 24), 
                            torch.randn(1, 2, 300, 300))
     
    test_set = OccNerfDataset(torch.randn(1, 18, 300, 300, 24), 
                            torch.randn(1, 18, 300, 300, 24), 
                            torch.randn(1, 2, 300, 300))

    return train_set, test_set
