import os
import shutil
import random
import pandas as pd
from glob import glob
from tqdm import tqdm

# --- Configuration ---
# Path to the raw dataset
RAW_DATA_DIR = os.path.join('data', 'raw') 

# Path where processed/balanced data will be saved
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Number of images to sample per style to balance the dataset
SAMPLES_PER_CLASS = 400

def balance_and_process_dataset():
    """
    Reads the raw dataset, samples a fixed number of images per class to balance the data,
    and saves the processed dataset along with a metadata CSV file.
    """
    # 1. Check if raw data exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found. Please check your path.")
        return

    # 2. Clean up the processed directory (remove if exists to start fresh)
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Cleaning existing processed directory: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(PROCESSED_DATA_DIR)

    # 3. Identify style folders (classes)
    # List directories only, ignoring hidden files or zips
    style_folders = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    print(f"Found {len(style_folders)} styles in raw data.")
    
    metadata = []
    
    # 4. Iterate over each style and sample images
    for style in tqdm(style_folders, desc="Processing Styles"):
        style_path = os.path.join(RAW_DATA_DIR, style)
        
        # Find all valid image files (jpg, jpeg, png)
        # Using glob to handle different extensions
        images = glob(os.path.join(style_path, '*.jpg')) + \
                 glob(os.path.join(style_path, '*.jpeg')) + \
                 glob(os.path.join(style_path, '*.png'))
        
        # Sampling Strategy: Undersampling
        if len(images) < SAMPLES_PER_CLASS:
            # If a class has fewer images than required, take all of them
            print(f"Warning: Style '{style}' has only {len(images)} images (Target: {SAMPLES_PER_CLASS}). All selected.")
            selected_images = images
        else:
            # Randomly sample the target number of images
            selected_images = random.sample(images, SAMPLES_PER_CLASS)
            
        # Create destination folder for this style
        dest_style_folder = os.path.join(PROCESSED_DATA_DIR, style)
        os.makedirs(dest_style_folder, exist_ok=True)
        
        # Copy selected images and record metadata
        for img_path in selected_images:
            file_name = os.path.basename(img_path)
            dest_path = os.path.join(dest_style_folder, file_name)
            
            # Copy file
            shutil.copy(img_path, dest_path)
            
            # Append to metadata list (relative path is better for portability)
            metadata.append({
                'image_path': os.path.join('data', 'processed', style, file_name),
                'style': style,
                'original_filename': file_name
            })
            
    # 5. Save Metadata to CSV
    # This CSV will be used by the Dataset class in PyTorch
    df = pd.DataFrame(metadata)
    csv_path = os.path.join('data', 'processed', 'metadata.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n--- Data Processing Complete ---")
    print(f"Total images processed: {len(df)}")
    print(f"Balanced dataset saved to: {PROCESSED_DATA_DIR}")
    print(f"Metadata CSV saved to: {csv_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility (Scientific requirement)
    random.seed(42)
    balance_and_process_dataset()