import os
import shutil
import random
import pandas as pd
from glob import glob
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
SAMPLES_PER_CLASS = 1000  # Changed per your request

# Folders to ignore (due to extra files in your structure)
IGNORE_FOLDERS = ['splits', 'raw', 'TrainingReady', '.ipynb_checkpoints']

def curate_dataset():
    print(f"🚀 Starting data processing (Target: {SAMPLES_PER_CLASS} images per style)...")
    
    # 1. Clean destination directory
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(PROCESSED_DATA_DIR)
    
    # 2. Identify styles
    all_items = os.listdir(RAW_DATA_DIR)
    style_folders = [d for d in all_items if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and d not in IGNORE_FOLDERS]
    
    print(f"✅ Styles found: {len(style_folders)}")
    metadata = []
    
    # 3. Process each style
    for style in tqdm(style_folders, desc="Processing Styles"):
        src_path = os.path.join(RAW_DATA_DIR, style)
        
        # Find all images
        images = glob(os.path.join(src_path, '*.jpg')) + \
                 glob(os.path.join(src_path, '*.jpeg')) + \
                 glob(os.path.join(src_path, '*.png'))
        
        # Sampling strategy
        if len(images) < SAMPLES_PER_CLASS:
            print(f"⚠️ Warning: Style '{style}' has only {len(images)} images. All selected.")
            selected_images = images
        else:
            selected_images = random.sample(images, SAMPLES_PER_CLASS)
            
        # Copy files
        dest_path = os.path.join(PROCESSED_DATA_DIR, style)
        os.makedirs(dest_path, exist_ok=True)
        
        for img_path in selected_images:
            filename = os.path.basename(img_path)
            # Physical file copy
            shutil.copy(img_path, os.path.join(dest_path, filename))
            
            # Save to metadata
            metadata.append({
                'image_path': os.path.join('data', 'processed', style, filename),
                'style': style,
                'original_filename': filename
            })
            
    # 4. Save final CSV
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n🎉 Step 1 complete. Total images: {len(df)}")
    print(f"Metadata file saved at: {csv_path}")

if __name__ == "__main__":
    random.seed(42)
    curate_dataset()