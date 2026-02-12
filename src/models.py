import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Baseline Convolutional Neural Network (CNN).
    
    Architecture:
    - 3 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
    - Flatten
    - Fully Connected Layers with Dropout
    
    Purpose:
    To serve as a performance benchmark for Phase 2.
    """
    def __init__(self, num_classes=27):
        super(SimpleCNN, self).__init__()
        
        # --- Block 1 ---
        # Input: (Batch, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output: (Batch, 32, 112, 112)

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Output after pool: (Batch, 64, 56, 56)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Output after pool: (Batch, 128, 28, 28)

        # --- Classifier ---
        # Flatten size: 128 channels * 28 * 28
        self.flatten_dim = 128 * 28 * 28
        
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting in baseline
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Feature Extractor
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Smoke Test: Check if model compiles and accepts input
    try:
        model = SimpleCNN(num_classes=10)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"✅ Model Test Passed. Output Shape: {output.shape}")
    except Exception as e:
        print(f"❌ Model Test Failed: {e}")