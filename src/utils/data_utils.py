import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

class RoundDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        
        self.data = np.load(filepath, mmap_mode='r')
        self.boards = self.data['boards']
        self.actions = self.data['actions']
        
        valid_mask = (self.actions >= 0) & (self.actions < 8100)
        self.valid_indices = np.where(valid_mask)[0]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        board = self.boards[original_idx]
        action = self.actions[original_idx]
        return torch.from_numpy(board).float(), torch.tensor(action, dtype=torch.long)


class FastImitationDataset(Dataset):
    def __init__(self, data_path, indices=None):
        self.data_path = data_path
        
        self.all_samples = []
        for filename in sorted(os.listdir(data_path)):
            if not filename.endswith('.npz'):
                continue
            filepath = os.path.join(data_path, filename)
            with np.load(filepath, mmap_mode='r') as data:
                actions = data['actions']
                valid_mask = (actions >= 0) & (actions < 8100)
                valid_indices = np.where(valid_mask)[0]
                for idx in valid_indices:
                    self.all_samples.append((filepath, int(idx)))
        
        if indices is None:
            self.indices = np.arange(len(self.all_samples))
        else:
            self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        filepath, local_idx = self.all_samples[global_idx]
        
        with np.load(filepath, mmap_mode='r') as data:
            board = data['boards'][local_idx]
            action = data['actions'][local_idx]
        
        return torch.from_numpy(board).float(), torch.tensor(action, dtype=torch.long)


def create_train_val_datasets(data_path, train_ratio=0.9):
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])
    
    train_files = all_files[:4]
    val_files = all_files[4:6]
    
    train_datasets = []
    for filename in train_files:
        filepath = os.path.join(data_path, filename)
        train_datasets.append(RoundDataset(filepath))
    
    val_datasets = []
    for filename in val_files:
        filepath = os.path.join(data_path, filename)
        val_datasets.append(RoundDataset(filepath))
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"训练集样本数: {len(train_dataset):,}")
    print(f"验证集样本数: {len(val_dataset):,}")
    
    return train_dataset, val_dataset