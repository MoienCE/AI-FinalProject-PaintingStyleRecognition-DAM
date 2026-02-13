import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.preprocessing.dataset import create_dataloaders
from src.models.models import get_model
from src.utils.utils import set_seed, save_metrics

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unfreeze_model_layers(model, model_type='resnet50'):
    """
    Unfreezes the last blocks of the model for Fine-Tuning.
    Adapts based on architecture (ResNet vs EfficientNet).
    """
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    if model_type == 'resnet50':
        # Unfreeze FC and Layer 4
        for param in model.base_model.fc.parameters():
            param.requires_grad = True
        for param in model.base_model.layer4.parameters():
            param.requires_grad = True
            
    elif model_type == 'efficientnet_b0':
        # Unfreeze Classifier
        for param in model.base_model.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze the last 2 blocks of features (Deepest layers)
        # EfficientNet features is a Sequential list of blocks.
        # We unfreeze the last 20% of the network.
        for param in model.base_model.features[-3:].parameters(): 
            param.requires_grad = True
            
    print(f"✅ Model ({model_type}) Unfrozen: Last layers are now trainable.")
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Fine-Tuning", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def main(args):
    set_seed(42)
    
    # 1. Setup Data (Balanced + Heavy Augmentation)
    print("⏳ Loading Data with Strong Augmentation...")
    dataloaders, datasets = create_dataloaders(
        splits_dir=os.path.join('data', 'processed', 'splits'),
        root_dir='.',
        batch_size=args.batch_size,
        use_sampler=True 
    )
    num_classes = len(datasets['train'].data['label'].unique())
    
    # 2. Load New Model Architecture
    print(f"🏗️ Initializing {args.model_type}...")
    model = get_model(args.model_type, num_classes=num_classes, device=DEVICE)
    
    # Note: We are NOT loading the old ResNet checkpoint because architecture changed.
    # We are starting transfer learning from ImageNet weights again, but with better strategies.
    
    # 3. Apply Unfreezing
    model = unfreeze_model_layers(model, args.model_type)
    
    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, 
                           weight_decay=0.01)
    
    # Cosine Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    writer = SummaryWriter(log_dir=os.path.join("results", "runs", f"{args.model_type}_{args.exp_name}"))
    
    # 5. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🔥 Starting Training ({args.model_type}) for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, DEVICE)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("models", f"best_model_{args.model_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🏆 Best Model Saved! (Acc: {best_acc:.4f})")
            
    writer.close()
    save_metrics(history, os.path.join("results", f"history_{args.model_type}.json"))
    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Change default to efficientnet_b0
    parser.add_argument('--model_type', type=str, default='efficientnet_b0') 
    parser.add_argument('--epochs', type=int, default=30) # EfficientNet takes time to converge
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4) # Slightly higher start LR for fresh transfer
    parser.add_argument('--exp_name', type=str, default='v5_efficientnet_balanced')
    
    args = parser.parse_args()
    main(args)