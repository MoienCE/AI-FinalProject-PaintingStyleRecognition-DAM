import os
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import sys

# Import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.models import get_model

# --- Config ---
app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = 'efficientnet_b3'
IMG_SIZE = 384

# --- 1. Fix Path Finding ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', f'best_model_{MODEL_TYPE}.pth')
MAPPING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'splits', 'class_mapping.json')

# --- 2. Robust Class Mapping Loading ---
idx_to_class = {}

if os.path.exists(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
            
        # اصلاح شده بر اساس فایل JSON شما:
        # ساختار فایل شما: {"0": "Abstract_Expressionism", ...}
        for index_str, style_name in mapping.items():
            # کلید (index_str) عدد است، مقدار (style_name) اسم است
            idx_to_class[int(index_str)] = style_name
            
        print(f"✅ Class mapping loaded successfully! ({len(idx_to_class)} classes)")
        # چاپ ۵ مورد اول برای اطمینان
        print(f"   Sample: {list(idx_to_class.values())[:5]}...")
        
    except Exception as e:
        print(f"❌ Error parsing mapping file: {e}")
else:
    print(f"⚠️ Warning: Mapping file NOT found at: {MAPPING_PATH}")
    print("   Run 'python src/preprocessing/preprocessing.py' to generate it.")

# --- Load Model ---
print(f"🏗️ Loading Model: {MODEL_TYPE}...")
num_classes = len(idx_to_class) if idx_to_class else 27
model = get_model(MODEL_TYPE, num_classes=num_classes, device=DEVICE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: Model path {MODEL_PATH} not found.")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Process Image
        image = Image.open(file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get Top 3 Predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            results = []
            for i in range(3):
                idx = top3_idx[i].item()
                prob = top3_prob[i].item()
                # دریافت نام کلاس از دیکشنری یا استفاده از ایندکس در صورت نبودن
                class_name = idx_to_class.get(idx, f"Class {idx}")
                results.append({
                    "style": class_name,
                    "confidence": round(prob * 100, 2)
                })
        
        return jsonify({
            "top_prediction": results[0],
            "alternatives": results[1:]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)