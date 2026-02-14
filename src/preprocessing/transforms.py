from torchvision import transforms

# ImageNet Statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 384

def get_transforms(stage='train'):
    """
    Returns transforms for a specific stage.
    Now includes TrivialAugmentWide for stronger regularization.
    """
    if stage == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), # Increased scale range
            transforms.RandomHorizontalFlip(p=0.5),
            
            # --- Heavy Augmentation Strategy ---
            # TrivialAugmentWide automatically applies optimal augmentations
            transforms.TrivialAugmentWide(), 
            
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            
            # Random Erasing: Randomly blacks out parts of the image
            # Forces the model to look at the whole picture, not just one part
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        # Validation / Test / Inference
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])