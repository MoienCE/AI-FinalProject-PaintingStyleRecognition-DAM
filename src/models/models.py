import torch.nn as nn
from torchvision import models
# Import EfficientNet B0 and B3
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """The baseline model from Phase 1."""
    def __init__(self, num_classes=27):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.flatten_dim = 128 * 28 * 28
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransferLearningModel(nn.Module):
    """
    Phase 2 Advanced Model: Supports ResNet50, EfficientNet-B0, and EfficientNet-B3.
    """
    def __init__(self, num_classes=27, model_name='resnet50', fine_tune=True):
        super(TransferLearningModel, self).__init__()
        
        if model_name == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            if not fine_tune:
                for param in self.base_model.parameters():
                    param.requires_grad = False
            
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif model_name == 'efficientnet_b0':
            self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            if not fine_tune:
                for param in self.base_model.parameters():
                    param.requires_grad = False
            
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif model_name == 'efficientnet_b3':
            # EfficientNet-B3: Higher resolution (300-384px) & Deeper
            self.base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            
            if not fine_tune:
                for param in self.base_model.parameters():
                    param.requires_grad = False
            
            # Classifier modification
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
    def forward(self, x):
        return self.base_model(x)

def get_model(model_type, num_classes=27, device='cpu'):
    if model_type == 'baseline':
        model = SimpleCNN(num_classes)
    elif model_type in ['resnet50', 'efficientnet_b0', 'efficientnet_b3']:
        model = TransferLearningModel(num_classes, model_name=model_type, fine_tune=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)