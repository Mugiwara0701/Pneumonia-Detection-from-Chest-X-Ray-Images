import os
import json
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

num_classes = 2
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model_path = './model/resnet34_lungs_disease.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

remedies_file = "class_names.json"

with open(remedies_file, "r") as f:
    remedies_info = json.load(f)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_condition(image_path):
    """Detect condition from the uploaded image."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        condition = remedies_info[predicted.item()]["condition"]
        remedies = remedies_info[predicted.item()]["remedies"]

    return {
        "condition": condition,
        "remedies": remedies
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict condition from an image."""
    if not file or file.filename == '':
        raise HTTPException(status_code=400, detail="No image file selected")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        result = detect_condition(file_path)

        os.remove(file_path)

        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
