import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb  

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
    """
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    if model_type == 'resnet50':
        for param in model.base_model.fc.parameters():
            param.requires_grad = True
        for param in model.base_model.layer4.parameters():
            param.requires_grad = True
            
    elif model_type in ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4']:
        # Unfreeze Classifier
        for param in model.base_model.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze the last 3 blocks of features (Deepest layers)
        for param in model.base_model.features[-3:].parameters(): 
            param.requires_grad = True
            
    print(f"✅ Model ({model_type}) Unfrozen: Last layers are now trainable.")
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Fine-Tuning", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def main(args):
    set_seed(42)
    
    # --- WandB Initialization ---
    wandb.init(
        project="Art-Style-Recognition",
        name=args.exp_name,
        config=vars(args)
    )
    
    # 1. Setup Data
    print(f"⏳ Loading Data (Batch Size: {args.batch_size})...")
    dataloaders, datasets = create_dataloaders(
        splits_dir=os.path.join('data', 'processed', 'splits'),
        root_dir='.',
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        use_sampler=True 
    )
    num_classes = len(datasets['train'].data['label'].unique())
    print(f"✅ Classes detected: {num_classes}")
    
    # 2. Load Model Architecture
    print(f"🏗️ Initializing {args.model_type}...")
    model = get_model(args.model_type, num_classes=num_classes, device=DEVICE)
    
    # 3. Apply Unfreezing
    model = unfreeze_model_layers(model, args.model_type)
    
    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    writer = SummaryWriter(log_dir=os.path.join("results", "runs", f"{args.model_type}_{args.exp_name}"))
    
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🔥 Starting Training ({args.model_type}) for {args.epochs} epochs...")
    
    # Log model gradients to wandb (Optional - adds overhead but cool visualization)
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, DEVICE)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # --- Logging to TensorBoard ---
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # --- Logging to WandB ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr
        })
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("models", f"best_model_{args.model_type}.pth")
            
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"🏆 Best Model Saved! (Acc: {best_acc:.4f})")
            
    writer.close()
    wandb.finish() # End WandB run
    
    os.makedirs("results", exist_ok=True)
    save_metrics(history, os.path.join("results", f"history_{args.model_type}.json"))
    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Defaulting to B3
    parser.add_argument('--model_type', type=str, default='efficientnet_b3') 
    parser.add_argument('--epochs', type=int, default=50) 
    
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--num_workers', type=int, default=4)  
    
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--exp_name', type=str, default='v7_b3_max_power_wandb')
    
    args = parser.parse_args()
    main(args)