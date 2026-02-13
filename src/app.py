import os
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import json

# Import your model modules
from models.models import get_model

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
# We will use the EfficientNet model once trained, strictly adhering to phase-2
MODEL_PATH = "models/best_model_efficientnet_b0.pth" 
CLASS_MAPPING_PATH = "data/processed/splits/class_mapping.json"

# --- Load Class Mapping ---
if os.path.exists(CLASS_MAPPING_PATH):
    with open(CLASS_MAPPING_PATH, 'r') as f:
        # Load and flip mapping to be {index: name} just in case
        raw_mapping = json.load(f)
        # If mapping is {name: index}, invert it. If {index: name}, keep it.
        # Based on previous steps, it was saved as {name: index} usually, let's ensure:
        idx_to_class = {v: k for k, v in raw_mapping.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    class_names = [f"Class {i}" for i in range(27)]

num_classes = len(class_names)

# --- Load Model ---
print("⏳ Loading Model for Demo...")
try:
    model = get_model('efficientnet_b0', num_classes=num_classes, device=DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded.")
    else:
        print("⚠️ Warning: Weight file not found. Using random weights.")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return None
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
    
    # Get top 3 predictions
    topk_probs, topk_indices = torch.topk(probs, 3)
    
    # Format for Gradio
    results = {}
    for i in range(3):
        idx = topk_indices[0][i].item()
        score = topk_probs[0][i].item()
        class_name = class_names[idx]
        results[class_name] = score
        
    return results

# --- Gradio Interface ---
title = "🎨 Art Style Recognizer (EfficientNet)"
description = "Upload a painting to identify its artistic style (e.g., Cubism, Impressionism)."
article = "Developed for AI Final Project - Phase 2"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Artwork"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title=title,
    description=description,
    article=article,
    theme="default"
)

if __name__ == "__main__":
    iface.launch(share=True)