import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Add project root to system path
sys.path.append(os.getcwd())

from src.preprocessing.dataset import create_dataloaders
from src.models.models import get_model

# Constants for Un-normalization (to visualize images correctly)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Directories
RESULTS_DIR = os.path.join("results", "metrics")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_checkpoint(model, checkpoint_path):
    """Loads the best model weights."""
    print(f"📥 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    return model

def unnormalize_image(tensor):
    """Converts a tensor back to a displayable image."""
    img = tensor.cpu().numpy().transpose((1, 2, 0)) # (C, H, W) -> (H, W, C)
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return img

def get_predictions_and_errors(model, dataloader, device):
    """
    Runs inference and collects metrics + error examples.
    """
    model.eval()
    all_preds = []
    all_labels = []
    error_samples = [] # To store (image, true_label, pred_label)
    
    print("🔎 Running Inference & Looking for Errors...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store Metrics
            current_preds = preds.cpu().numpy()
            current_labels = labels.numpy()
            all_preds.extend(current_preds)
            all_labels.extend(current_labels)
            
            # Find Errors in this batch
            incorrect_indices = np.where(current_preds != current_labels)[0]
            for idx in incorrect_indices:
                if len(error_samples) < 16: # Keep top 16 errors for visualization
                    img = inputs[idx]
                    true_lbl = current_labels[idx]
                    pred_lbl = current_preds[idx]
                    error_samples.append((img, true_lbl, pred_lbl))

    return np.array(all_labels), np.array(all_preds), error_samples

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plots and saves the confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path)
    print(f"📊 Confusion Matrix saved to {save_path}")

def visualize_errors(error_samples, class_names, model_name):
    """
    Visualizes False Positives/Negatives (Requirement: image_5c8aa2.png).
    """
    if not error_samples:
        print("🎉 Amazing! No errors found to visualize.")
        return

    plt.figure(figsize=(16, 16))
    plt.suptitle(f"Error Analysis: False Predictions ({model_name})", fontsize=20)
    
    rows = int(np.ceil(np.sqrt(len(error_samples))))
    cols = int(np.ceil(len(error_samples) / rows))
    
    for i, (img_tensor, true_idx, pred_idx) in enumerate(error_samples):
        img = unnormalize_image(img_tensor)
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_name}\nPred: {pred_name}", color='red', fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'error_analysis_{model_name}.png')
    plt.savefig(save_path)
    print(f"🖼️ Error Analysis Chart saved to {save_path}")

def save_classification_report(y_true, y_pred, class_names, model_name):
    """Saves detailed classification metrics (Precision, Recall, F1)."""
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\n" + "="*60)
    print(f"🧪 Classification Report for {model_name}")
    print("="*60)
    print(report_str)
    
    save_path = os.path.join(RESULTS_DIR, f'evaluation_report_{model_name}.json')
    with open(save_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"📄 Metrics JSON saved to {save_path}")

def main(model_type='resnet50'):
    # 1. Load Data
    dataloaders, datasets = create_dataloaders(
        splits_dir=os.path.join('data', 'processed', 'splits'),
        root_dir='.',
        batch_size=32
    )
    
    # Load Class Mapping
    mapping_path = os.path.join('data', 'processed', 'splits', 'class_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            mapping = json.load(f)
            # Ensure sorting by index
            class_names = [mapping[str(i)] for i in range(len(mapping))]
    else:
        print("⚠️ Warning: class_mapping.json not found. Using numeric labels.")
        class_names = [str(i) for i in range(27)]

    num_classes = len(class_names)

    # 2. Initialize Model
    model = get_model(model_type, num_classes=num_classes, device=DEVICE)
    
    # 3. Load Best Weights
    checkpoint_path = os.path.join("models", f"best_model_{model_type}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        return

    model = load_checkpoint(model, checkpoint_path)

    # 4. Run Evaluation
    y_true, y_pred, error_samples = get_predictions_and_errors(model, dataloaders['test'], DEVICE)

    # 5. Generate Outputs
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    
    save_classification_report(y_true, y_pred, class_names, model_type)
    plot_confusion_matrix(y_true, y_pred, class_names, model_type)
    visualize_errors(error_samples, class_names, model_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet50', choices=['baseline', 'resnet50'])
    args = parser.parse_args()
    main(args.model_type)