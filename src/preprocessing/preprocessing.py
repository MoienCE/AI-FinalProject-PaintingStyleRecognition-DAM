import os
import json
import pandas as pd
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration Constants ---
# Based on EDA findings and ImageNet standards
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def prepare_splits(metadata_path, output_dir, seed=42):
    """
    Phase 1 - Step 3: Data Splitting Strategy
    
    Reads the metadata CSV, encodes labels, and performs a stratified split
    into Train (70%), Validation (15%), and Test (15%) sets.
    
    Args:
        metadata_path (str): Path to the balanced metadata CSV.
        output_dir (str): Directory to save the split CSVs and mapping.
        seed (int): Random seed for reproducibility.
    """
    print(f"⚙️ Starting Preprocessing: Reading from {metadata_path}...")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found! Please run Step 1 (data_curation) first.")
        
    df = pd.read_csv(metadata_path)
    
    # --- 1. Label Encoding ---
    # Convert string labels (e.g., 'Cubism') to integers (0, 1, ...)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['style'])
    
    # Save the mapping (ID -> Style Name) for future inference/demo
    label_mapping = {int(k): v for k, v in zip(le.transform(le.classes_), le.classes_)}
    
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)
        
    print(f"✅ Label Encoding complete. Mapping saved to {mapping_path}")
    
    # --- 2. Stratified Splitting ---
    # We use 'stratify' to ensure minority classes are represented in all sets
    
    # Step A: Split Total -> Train (70%) + Temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['label'], 
        random_state=seed
    )
    
    # Step B: Split Temp -> Val (15%) + Test (15%)
    # Splitting the 30% exactly in half gives 15% of the total each
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['label'], 
        random_state=seed
    )
    
    # --- 3. Save Splits ---
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n📊 Data Split Statistics:")
    print(f"  - Train Set: {len(train_df)} images (70%) -> Saved to {train_path}")
    print(f"  - Val Set:   {len(val_df)} images (15%) -> Saved to {val_path}")
    print(f"  - Test Set:  {len(test_df)} images (15%) -> Saved to {test_path}")

def get_transforms(stage='train'):
    """
    Returns the PyTorch transform pipeline for a given stage.
    
    Args:
        stage (str): 'train', 'val', or 'test'.
    
    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if stage == 'train':
        return transforms.Compose([
            # EDA showed varying aspect ratios. RandomResizedCrop is better than simple Resize
            # as it prevents distortion and acts as augmentation.
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            
            # Data Augmentation to prevent Overfitting
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            # Standard conversion
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
    else: # val or test
        return transforms.Compose([
            # For validation, we just want to resize deterministically
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

if __name__ == "__main__":
    # Define paths based on project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level
    METADATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'metadata.csv')
    OUTPUT_SPLITS = os.path.join(BASE_DIR, 'data', 'processed', 'splits')
    
    try:
        prepare_splits(METADATA_FILE, OUTPUT_SPLITS)
    except Exception as e:
        print(f"❌ Error: {e}")