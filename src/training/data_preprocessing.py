import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(metadata_path, output_dir, seed=42):
    """
    Phase 1 - Step 3: Data Preprocessing
    
    Responsibilities:
    1. Label Encoding: Converts string labels (styles) to integers.
    2. Stratified Splitting: Splits data into Train (70%), Val (15%), Test (15%)
       while maintaining the class distribution found in EDA.
    """
    print("Starting Data Preprocessing (Phase 1 Step 3)...")
    
    # 1. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Data Loaded. Total samples: {len(df)}")

    # 2. Label Encoding (Style -> Integer)
    # This addresses the requirement to handle categorical data
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['style'])
    
    # Save the mapping (e.g., 0: 'Abstract', 1: 'Baroque'...) for later inference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    label_mapping = {k: int(v) for k, v in label_mapping.items()} # Convert to python int
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)
    print(f"Class mapping saved. Total classes: {len(label_mapping)}")

    # 3. Stratified Splitting (70% Train / 15% Val / 15% Test)
    # Using 'stratify' is crucial because EDA showed class imbalance (e.g., Action Painting has fewer samples)
    
    # Step A: Split Total into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30,  # 30% remaining for Val + Test
        stratify=df['label'], 
        random_state=seed
    )

    # Step B: Split Temp (30%) exactly in half -> 15% Val, 15% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, # 50% of 30% = 15% of total
        stratify=temp_df['label'], 
        random_state=seed
    )

    # 4. Save Splits as CSV manifests
    # We do NOT save images here, only paths. This keeps the repo light.
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("\nSplit Statistics (Target: 70/15/15):")
    total = len(df)
    print(f"   - Train Set: {len(train_df)} images ({len(train_df)/total:.1%})")
    print(f"   - Val Set:   {len(val_df)} images ({len(val_df)/total:.1%})")
    print(f"   - Test Set:  {len(test_df)} images ({len(test_df)/total:.1%})")
    
    # Validation Check
    if len(train_df) + len(val_df) + len(test_df) != total:
        print("Warning: Sum of splits does not match total!")
    else:
        print(f"\nSplitting Successful! Files saved to: {output_dir}")

if __name__ == "__main__":
    METADATA_PATH = os.path.join('data', 'processed', 'metadata.csv')
    OUTPUT_DIR = os.path.join('data', 'processed', 'splits')
    
    try:
        preprocess_data(METADATA_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"Error: {e}")