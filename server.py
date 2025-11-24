# ==========================================
# server.py — Druk Health CTG AI Backend (Final Updated Version)
# ==========================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
import tempfile
import joblib
import cv2
import os
from scipy.signal import find_peaks, medfilt
from collections import Counter
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# ------------------------------
# Load environment
# ------------------------------
load_dotenv()

# ------------------------------
# Client-facing label converter
# ------------------------------
def convert_to_client_label(clinical_label):
    mapping = {
        "Normal": "Reassuring",
        "Suspect": "Non-Reassuring",
        "Pathological": "Abnormal"
    }
    return mapping.get(clinical_label, clinical_label)

# ------------------------------
# Clinical-aware model wrapper
# ------------------------------
class ClinicalAwareCTGModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.class_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}

    def clinical_class(self, baseline, decels):
        """Medical rule-based classification"""
        
        # PATHOLOGICAL
        if baseline < 100 or baseline > 180:
            return "Pathological"
        if decels > 50:
            return "Pathological"

        # SUSPECT
        if 100 <= baseline < 110 or 160 < baseline <= 180:
            return "Suspect"
        if 1 <= decels <= 50:
            return "Suspect"

        # NORMAL
        if 110 <= baseline <= 160 and decels == 0:
            return "Normal"

        return "Suspect"

    def predict(self, X):
        baseline = X["baseline value"].values[0]
        decels = X["prolongued_decelerations"].values[0]

        clinical = self.clinical_class(baseline, decels)

        # If clinically Pathological → override model
        if clinical == "Pathological":
            return "Pathological"

        # Otherwise use model
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        model_label = self.class_map.get(pred, "Suspect")

        # If clinical says Suspect but model says Normal → keep Suspect
        if clinical == "Suspect" and model_label == "Normal":
            return "Suspect"

        return model_label

# ------------------------------
# FastAPI initialization
# ------------------------------
app = FastAPI(title="Druk Health CTG AI Backend")

origins = [
    "http://localhost:5173",
    "https://drukhealthfrontend.vercel.app",
    "https://fastapi-backend-yrc0.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# MongoDB connection
# ------------------------------
MONGO_URI = "mongodb+srv://12220045gcit:Kunzang1234@cluster0.rskaemg.mongodb.net/drukhealth?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["drukhealth"]
ctg_collection = db["ctgscans"]

# ------------------------------
# Cloudinary config
# ------------------------------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "522272821951884"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")
)

# ------------------------------
# Load model & scaler
# ------------------------------
model, scaler = joblib.load("clinical_aware_ctg_model .pkl")
model_ctg_class = ClinicalAwareCTGModel(model, scaler)
print("✅ Model & scaler loaded successfully")

# =======================================================
# Signal extraction helpers (IMPROVED)
# =======================================================
def extract_trace_from_image_gray(trace_img):
    blurred = medfilt(trace_img, kernel_size=5)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = 255 - th

    h, w = th.shape
    ys = []
    for x in range(w):
        idx = np.where(th[:, x] > 127)[0]
        ys.append(np.median(idx) if len(idx) else np.nan)

    ys = pd.Series(ys).interpolate(limit_direction="both").values
    ys = h - ys
    ys = medfilt(ys, kernel_size=5)

    return ys.astype(float)

def extract_ctg_signals(image_path, fhr_range=(50, 210)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 127:
        gray = 255 - gray

    h, w = gray.shape
    fhr_img = gray[:int(0.55*h), :]
    uc_img = gray[int(0.55*h):, :]

    fhr_pixels = extract_trace_from_image_gray(fhr_img)
    uc_pixels = extract_trace_from_image_gray(uc_img)

    px_min = np.nanmin(fhr_pixels)
    px_max = np.nanmax(fhr_pixels)
    denom = px_max - px_min if px_max != px_min else 1

    min_val, max_val = fhr_range
    fhr_signal = min_val + (fhr_pixels - px_min) * (max_val - min_val) / denom

    return fhr_signal, uc_pixels, 4  # assume 4 Hz

# =======================================================
# Deceleration detection (FINAL VERSION)
# =======================================================
def detect_decelerations(signal, baseline, sampling_rate):

    min_drop = 15
    min_duration = int(15 * sampling_rate)  # 15 s

    decel_count = 0
    i = 0
    N = len(signal)

    while i < N:
        if signal[i] < baseline - min_drop:
            start = i
            while i < N and signal[i] < baseline - 5:
                i += 1
            if i - start >= min_duration:
                decel_count += 1
        else:
            i += 1

    return decel_count

# =======================================================
# Feature extraction
# =======================================================
def compute_features(fhr_signal, uc_signal, sampling_rate):

    window = int(60 * sampling_rate)  # 1 minute
    baselines = [np.mean(fhr_signal[i:i+window]) for i in range(0, len(fhr_signal), window)
                 if i+window <= len(fhr_signal)]
    baseline = float(np.mean(baselines))

    total_sec = len(fhr_signal) / sampling_rate
    total_min = total_sec / 60
    factor_10 = total_min / 10 if total_min > 0 else 1

    accel_count = detect_accelerations(fhr_signal, baseline, sampling_rate)
    decels = detect_decelerations(fhr_signal, baseline, sampling_rate)

    fetal_movement = int(np.sum(np.diff(uc_signal) > 10) / factor_10)
    uc_peaks, _ = find_peaks(uc_signal, height=np.nanmean(uc_signal) + 10)
    uterine_contractions = int(len(uc_peaks) / factor_10)

    return {
        "baseline value": baseline,
        "accelerations": int(accel_count),
        "fetal_movement": int(fetal_movement),
        "uterine_contractions": int(uterine_contractions),
        "prolongued_decelerations": int(decels)
    }

def detect_accelerations(signal, baseline, sampling_rate):

    min_rise = 15
    min_duration = int(15 * sampling_rate)

    accels = 0
    i = 0
    N = len(signal)

    while i < N:
        if signal[i] > baseline + min_rise:
            start = i
            while i < N and signal[i] > baseline + 5:
                i += 1
            if i - start >= min_duration:
                accels += 1
        else:
            i += 1

    return accels

# =======================================================
# ROUTES
# =======================================================
@app.post("/predict/")
async def predict_ctg(file: UploadFile = File(...)):

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        path = tmp.name

    try:
        fhr_signal, uc_signal, fs = extract_ctg_signals(path)
        features = compute_features(fhr_signal, uc_signal, fs)

        df = pd.DataFrame([features])

        clinical_label = model_ctg_class.predict(df)
        client_label = convert_to_client_label(clinical_label)

        # Upload to Cloudinary
        try:
            upload_result = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
            img_url = upload_result.get("secure_url")
        except:
            img_url = None

        record = {
            "timestamp": datetime.utcnow() + timedelta(hours=6),
            "ctgDetected": clinical_label,
            "clientLabel": client_label,
            "features": features,
            "imageUrl": img_url,
        }

        result = ctg_collection.insert_one(record)

        return {
            "label": client_label,
            "clinical_label": clinical_label,
            "features": features,
            "imageUrl": img_url,
            "record_id": str(result.inserted_id),
        }

    finally:
        os.remove(path)

@app.get("/records")
def records():
    recs = list(ctg_collection.find().sort("timestamp", -1))
    return {"records": [
        {
            "id": str(r["_id"]),
            "timestamp": r.get("timestamp"),
            "ctgDetected": r.get("ctgDetected"),
            "clientLabel": r.get("clientLabel"),
            "features": r.get("features"),
            "imageUrl": r.get("imageUrl")
        }
        for r in recs
    ]}

@app.delete("/records/{record_id}")
def delete_record(record_id: str):
    r = ctg_collection.delete_one({"_id": ObjectId(record_id)})
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"detail": "Record deleted"}
@app.get("/api/analysis")
def get_analysis():
    records = list(ctg_collection.find({}, {"_id": 0}))
    if not records:
        return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

    df = pd.DataFrame(records)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Time series summary
    pivot = df.pivot_table(
        index="date",
        columns="ctgDetected",
        aggfunc="size",
        fill_value=0
    ).reset_index()

    pivot = pivot.rename(columns={
        "Normal": "N",
        "Suspect": "S",
        "Pathologic": "P"
    })

    time_series = pivot.to_dict(orient="records")

    # Total NSP counts
    from collections import Counter
    counts = Counter(df["ctgDetected"])

    nspStats = {
        "Normal": int(counts.get("Normal", 0)),
        "Suspect": int(counts.get("Suspect", 0)),
        "Pathologic": int(counts.get("Pathologic", 0)),
    }

    return {
        "predictions": time_series,
        "nspStats": nspStats
    }

# Run
# =======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
