import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.preprocessing.transforms import get_transforms

class ArtDataset(Dataset):
    def __init__(self, csv_file, root_dir, stage='train'):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = get_transforms(stage)
        self.labels = self.data['label'].tolist() # Needed for sampling
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_rel_path = row['image_path']
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
            return self.__getitem__((idx - 1) % len(self))
            
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_sampler(dataset):
    """
    Calculates weights for WeightedRandomSampler to handle class imbalance.
    Classes with fewer samples get higher weights.
    """
    targets = dataset.labels
    class_counts = pd.Series(targets).value_counts().sort_index()
    
    # Calculate weight per class: 1 / count
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
    
    # Assign weight to each sample
    sample_weights = [class_weights[t] for t in targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def create_dataloaders(splits_dir, root_dir, batch_size=32, num_workers=4, use_sampler=False):
    """
    Creates train, val, and test dataloaders.
    """
    datasets = {
        x: ArtDataset(
            csv_file=os.path.join(splits_dir, f'{x}.csv'),
            root_dir=root_dir,
            stage=x
        ) for x in ['train', 'val', 'test']
    }
    
    # Setup Training Loader (with optional Sampler for imbalance)
    if use_sampler:
        print("⚖️ Using WeightedRandomSampler to fix Class Imbalance.")
        train_sampler = get_sampler(datasets['train'])
        train_shuffle = False # Shuffle is mutually exclusive with sampler
    else:
        train_sampler = None
        train_shuffle = True
        
    use_persistent = True if num_workers > 0 else False

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, 
                          shuffle=train_shuffle, sampler=train_sampler, 
                          num_workers=num_workers, pin_memory=True, 
                          persistent_workers=use_persistent),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=True, 
                        persistent_workers=use_persistent),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, pin_memory=True, 
                         persistent_workers=use_persistent)
    }
    
    return dataloaders, datasets