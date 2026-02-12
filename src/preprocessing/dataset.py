import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# Import transforms from our new module
from src.preprocessing.transforms import get_transforms

class ArtDataset(Dataset):
    def __init__(self, csv_file, root_dir, stage='train'):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Project root directory to handle relative paths.
            stage (str): 'train', 'val', or 'test' to select transforms.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = get_transforms(stage)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Adjust path: remove 'data/' if it's already in root, or handle relative paths
        # Assuming metadata paths are like "data/processed/Style/img.jpg"
        img_rel_path = row['image_path']
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback to previous image to prevent crash
            return self.__getitem__((idx - 1) % len(self))
            
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(splits_dir, root_dir, batch_size=32, num_workers=2):
    """Creates train, val, and test dataloaders."""
    
    datasets = {
        x: ArtDataset(
            csv_file=os.path.join(splits_dir, f'{x}.csv'),
            root_dir=root_dir,
            stage=x
        ) for x in ['train', 'val', 'test']
    }
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }
    
    return dataloaders, datasets