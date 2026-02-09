import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ArtDataset(Dataset):
    """
    Custom Dataset for loading Art images.
    Reads paths and labels from the processed CSV files (train/val/test).
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get row data
        row = self.data_frame.iloc[idx]
        
        # Construct image path using column name 'image_path' (Safer than index)
        img_rel_path = row['image_path'] 
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            print(f"Warning: Skipping corrupted image: {img_path}")
            # Return the next image instead to avoid crash
            return self.__getitem__((idx + 1) % len(self))

        # Get label using column name 'label'
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=2):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    
    # ImageNet Standard Mean and Std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Training Transforms (with Augmentation placeholders)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),        
        transforms.CenterCrop(224),           
        # transforms.RandomHorizontalFlip(),  # Uncomment for Phase 2
        transforms.ToTensor(),
        normalize
    ])

    # Validation/Test Transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # CSV Paths
    splits_dir = os.path.join(data_dir, 'processed', 'splits')
    
    # Datasets
    train_dataset = ArtDataset(
        csv_file=os.path.join(splits_dir, 'train.csv'),
        root_dir='.', 
        transform=train_transform
    )
    
    val_dataset = ArtDataset(
        csv_file=os.path.join(splits_dir, 'val.csv'),
        root_dir='.',
        transform=val_transform
    )
    
    test_dataset = ArtDataset(
        csv_file=os.path.join(splits_dir, 'test.csv'),
        root_dir='.',
        transform=val_transform
    )

    # DataLoaders (num_workers=0 for Windows safety)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"DataLoaders Created:")
    print(f"   - Train Batches: {len(train_loader)}")
    print(f"   - Val Batches:   {len(val_loader)}")
    print(f"   - Test Batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the loader independently
    loaders = get_data_loaders(data_dir='data', batch_size=16)
    
    # Fetch one batch to verify
    try:
        images, labels = next(iter(loaders[0]))
        print(f"\nBatch Verification:")
        print(f"   - Image Batch Shape: {images.shape} (Batch, Channel, Height, Width)")
        print(f"   - Label Batch Shape: {labels.shape}")
        print("   - Sample Labels:", labels[:5].tolist())
    except Exception as e:
        print(f"\nTest Failed: {e}")