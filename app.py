from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pickle
import json
import os
import time
import uuid
import base64
import numpy as np
import cv2

# Global variables
references = None
train_features = None
train_labels = None
metrics = None
k_value = 19 # default optimal K

def extract_color_histogram(img_resized):
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    hist_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    sum_val = hist_color.sum()
    if sum_val > 0:
        hist_color /= sum_val
    return hist_color

def get_base64_thumbnail(img_path):
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to small thumbnail (e.g. 64x64) to minimize payload size
            thumb = cv2.resize(img, (64, 64))
            _, buffer = cv2.imencode('.jpg', thumb)
            encoded = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding thumbnail for {img_path}: {e}")
    return ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global references, train_features, train_labels, metrics, k_value
    
    # Paths for model artifacts in current directory
    data_path = "training_data.pkl"
    metrics_path = "metrics.json"
        
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            training_data = pickle.load(f)
        references = training_data["references"]
        train_features = np.array([ref["feature"] for ref in references])
        train_labels = np.array([ref["label"] for ref in references])
        print("Training reference data loaded successfully in pure NumPy format.")
    else:
        print(f"Warning: {data_path} not found.")
        
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        k_value = metrics.get("k_optimal", 19)
        print(f"Metrics loaded successfully. K-Optimal set to {k_value}.")
    else:
        print(f"Warning: {metrics_path} not found.")
        
    yield

app = FastAPI(
    title="WasteSort KNN Classification Engine (NumPy-Optimized)",
    description="Python backend API providing waste classification using a high-performance, pure NumPy KNN engine.",
    lifespan=lifespan
)

# Enable CORS for Next.js UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    model_status = "Loaded" if train_features is not None else "Not Loaded"
    return {
        "app": "WasteSort KNN Engine (NumPy-Optimized)",
        "status": "Online",
        "model_status": model_status,
        "k_value": k_value,
        "features_supported": ["Color Histogram"]
    }

@app.get("/api/metrics")
async def get_metrics():
    global metrics
    if metrics is None:
        metrics_path = "metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
        raise HTTPException(
            status_code=503, 
            detail="Model metrics are not available. Please run train.py first."
        )
    return metrics

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    global references, train_features, train_labels, k_value
    
    # Reload references if they weren't loaded on startup
    if references is None or train_features is None or train_labels is None:
        data_path = "training_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                training_data = pickle.load(f)
            references = training_data["references"]
            train_features = np.array([ref["feature"] for ref in references])
            train_labels = np.array([ref["label"] for ref in references])
        else:
            raise HTTPException(
                status_code=503, 
                detail="Reference data is not loaded. Please run train.py first."
            )
            
    t0 = time.time()
    
    try:
        # 1. Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
            
        # 2. Preprocessing & resizing
        img_resized = cv2.resize(img, (128, 128))
        
        # 3. Feature extraction (Color Histogram)
        feat = extract_color_histogram(img_resized)
        
        # 4. Pure NumPy Euclidean Distance calculation
        diff = train_features - feat
        distances = np.linalg.norm(diff, axis=1)
        
        # 5. Get K-nearest neighbors indices
        nearest_indices = np.argsort(distances)[:k_value]
        
        # 6. KNN voting
        neighbor_labels = train_labels[nearest_indices]
        organik_votes = int(np.sum(neighbor_labels == "Organik"))
        non_organik_votes = int(np.sum(neighbor_labels == "Non-Organik"))
        
        if organik_votes >= non_organik_votes:
            pred_label = "Organik"
            confidence = float(organik_votes / k_value)
        else:
            pred_label = "Non-Organik"
            confidence = float(non_organik_votes / k_value)
            
        # 7. Retrieve K-nearest neighbors details
        neighbors = []
        for rank_idx, idx in enumerate(nearest_indices):
            ref = references[idx]
            dist = distances[idx]
            rank = rank_idx + 1
            
            # Generate thumbnail encoded in base64
            thumb_b64 = get_base64_thumbnail(ref["path"])
            
            neighbors.append({
                "rank": rank,
                "label": ref["label"],
                "category": ref["category"],
                "distance": float(round(dist, 4)),
                "thumbnail_url": thumb_b64
            })
            
        execution_time = time.time() - t0
        
        # 8. Base64 encode the resized input image to return to client
        _, input_buf = cv2.imencode('.jpg', img_resized)
        input_b64 = f"data:image/jpeg;base64,{base64.b64encode(input_buf).decode('utf-8')}"
        
        return {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "prediction": pred_label,
            "confidence": confidence,
            "k_value": k_value,
            "neighbors": neighbors,
            "original_image_url": input_b64,
            "preprocessing": {
                "resized_to": [128, 128],
                "normalized": True
            },
            "features_used": ["Color Histogram"],
            "execution_time_seconds": float(round(execution_time, 4)),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
