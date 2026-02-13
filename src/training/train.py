import os
import sys
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# افزودن مسیر روت پروژه به سیستم تا پکیج src شناسایی شود
sys.path.append(os.getcwd())

# --- ایمپورت‌های اصلاح شده طبق ساختار پوشه‌بندی جدید ---
from src.preprocessing.dataset import create_dataloaders
from src.models.models import get_model
from src.utils.utils import set_seed, save_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- کلاس Early Stopping (طبق الزام image_5c8afd.png) ---
class EarlyStopping:
    """Stops training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, model, path):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path):
        if self.verbose:
            print(f'   🏆 Validation acc increased ({self.best_score:.4f} --> {val_acc:.4f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.best_score = val_acc

# --- توابع حلقه آموزش ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
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

# --- بدنه اصلی ---
def main(args):
    # 1. تنظیمات اولیه
    set_seed(42)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 2. راه اندازی TensorBoard (الزام image_b01cbb.png)
    writer = SummaryWriter(log_dir=os.path.join("results", "runs", f"{args.model_type}_{args.exp_name}"))
    
    # 3. بارگذاری داده
    print("⏳ Loading Data...")
    dataloaders, datasets = create_dataloaders(
        splits_dir=os.path.join('data', 'processed', 'splits'),
        root_dir='.', 
        batch_size=args.batch_size
    )
    # تشخیص خودکار تعداد کلاس‌ها
    num_classes = len(datasets['train'].data['label'].unique())
    print(f"✅ Data Loaded. Classes detected: {num_classes}")

    # 4. ساخت مدل
    print(f"🏗️ Initializing Model: {args.model_type}")
    model = get_model(args.model_type, num_classes=num_classes, device=DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # 5. تنظیم Optimizer و Scheduler
    if args.model_type == 'resnet50':
        # برای Transfer Learning معمولا لرنینگ ریت پایین‌تر بهتر است
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    # استفاده از Scheduler (یکی از روش‌های بهبود طبق image_5c8adb.png)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    # راه اندازی Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 6. حلقه آموزش
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🔥 Starting Training on {DEVICE}...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train & Val Steps
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, DEVICE)
        
        # Update Scheduler
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        
        # Logging
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # Write to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early Stopping & Checkpoint
        save_path = os.path.join(args.checkpoint_dir, f"best_model_{args.model_type}.pth")
        early_stopping(val_acc, model, save_path)
        
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break
            
    # پایان آموزش
    time_elapsed = time.time() - start_time
    print(f"\n🏁 Training Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {early_stopping.best_score:.4f}")
    
    # ذخیره فایل هیستوری برای رسم نمودار در گام بعدی
    save_metrics(history, os.path.join("results", f"history_{args.model_type}.json"))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet50', choices=['baseline', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--exp_name', type=str, default='v1')
    parser.add_argument('--checkpoint_dir', type=str, default='models')
    
    args = parser.parse_args()
    main(args)