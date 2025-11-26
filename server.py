
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from datetime import datetime, timedelta
# from pymongo import MongoClient
# from bson import ObjectId
# import pandas as pd
# import numpy as np
# import tempfile
# import joblib
# import cv2
# import os
# from scipy.signal import find_peaks, medfilt
# from collections import Counter
# import cloudinary
# import cloudinary.uploader
# from dotenv import load_dotenv

# # TensorFlow for CTG vs Non-CTG binary classifier
# import tensorflow as tf

# # ------------------------------
# # Load environment
# # ------------------------------
# load_dotenv()

# # ------------------------------
# # Client-facing label converter
# # ------------------------------
# def convert_to_client_label(clinical_label):
#     mapping = {
#         "Normal": "Reassuring",
#         "Suspect": "Non-Reassuring",
#         "Pathological": "Abnormal"
#     }
#     return mapping.get(clinical_label, clinical_label)

# # ------------------------------
# # Clinical-aware model wrapper
# # ------------------------------
# class ClinicalAwareCTGModel:
#     def __init__(self, model, scaler):  # <-- fix here
#         self.model = model
#         self.scaler = scaler
#         self.class_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}

#     def clinical_class(self, baseline, decels):
#         if baseline < 100 or baseline > 180:
#             return "Pathological"
#         if decels > 50:
#             return "Pathological"

#         if 100 <= baseline < 110 or 160 < baseline <= 180:
#             return "Suspect"
#         if 1 <= decels <= 50:
#             return "Suspect"

#         if 110 <= baseline <= 160 and decels == 0:
#             return "Normal"

#         return "Suspect"

#     def predict(self, X):
#         baseline = X["baseline value"].values[0]
#         decels = X["prolongued_decelerations"].values[0]

#         clinical = self.clinical_class(baseline, decels)

#         if clinical == "Pathological":
#             return "Pathological"

#         X_scaled = self.scaler.transform(X)
#         pred = self.model.predict(X_scaled)[0]
#         model_label = self.class_map.get(pred, "Suspect")

#         if clinical == "Suspect" and model_label == "Normal":
#             return "Suspect"

#         return model_label

# # ------------------------------
# # FastAPI initialization
# # ------------------------------
# app = FastAPI(title="Druk Health CTG AI Backend")

# origins = [
#     "http://localhost:5173",
#     "https://drukhealthfrontend.vercel.app",
#     "https://fastapi-backend-yrc0.onrender.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------------------
# # MongoDB connection
# # ------------------------------
# MONGO_URI = "mongodb+srv://12220045gcit:Kunzang1234@cluster0.rskaemg.mongodb.net/drukhealth?retryWrites=true&w=majority"

# client = MongoClient(MONGO_URI)
# db = client["drukhealth"]
# ctg_collection = db["ctgscans"]

# # ------------------------------
# # Cloudinary config
# # ------------------------------
# cloudinary.config(
#     cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b"),
#     api_key=os.getenv("CLOUDINARY_API_KEY", "522272821951884"),
#     api_secret=os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")
# )

# # ------------------------------
# # Load clinical CTG model & scaler
# # ------------------------------
# model, scaler = joblib.load("clinical_aware_ctg_model .pkl")
# model_ctg_class = ClinicalAwareCTGModel(model, scaler)
# print("✅ Clinical model loaded successfully")

# # ------------------------------
# # Load Binary CTG vs Non-CTG Model
# # ------------------------------
# binary_ctg_model = tf.keras.models.load_model("CTG_vs_NonCTG(1).keras")
# print("✅ Binary CTG vs Non-CTG CNN model loaded successfully")

# # ----------------------------------------------------------
# # Binary Classification Helper
# # ----------------------------------------------------------
# def classify_ctg_or_nonctg(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     pred = binary_ctg_model.predict(img_array)[0][0]

#     if pred > 0.5:
#         return "CTG", float(pred)
#     else:
#         return "Non-CTG", float(pred)

# # =======================================================
# # Signal extraction helpers
# # =======================================================
# def extract_trace_from_image_gray(trace_img):
#     blurred = medfilt(trace_img, kernel_size=5)
#     _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     if np.mean(th) < 127:
#         th = 255 - th
#     h, w = th.shape
#     ys = []
#     for x in range(w):
#         idx = np.where(th[:, x] > 127)[0]
#         ys.append(np.median(idx) if len(idx) else np.nan)
#     ys = pd.Series(ys).interpolate(limit_direction="both").values
#     ys = h - ys
#     ys = medfilt(ys, kernel_size=5)
#     return ys.astype(float)

# def extract_ctg_signals(image_path, fhr_range=(50, 210)):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError("Cannot read image")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if gray.mean() > 127:
#         gray = 255 - gray
#     h, w = gray.shape
#     fhr_img = gray[:int(0.55*h), :]
#     uc_img = gray[int(0.55*h):, :]
#     fhr_pixels = extract_trace_from_image_gray(fhr_img)
#     uc_pixels = extract_trace_from_image_gray(uc_img)

#     px_min = np.nanmin(fhr_pixels)
#     px_max = np.nanmax(fhr_pixels)
#     denom = px_max - px_min if px_max != px_min else 1

#     min_val, max_val = fhr_range
#     fhr_signal = min_val + (fhr_pixels - px_min) * (max_val - min_val) / denom

#     return fhr_signal, uc_pixels, 4

# # =======================================================
# # Deceleration detection
# # =======================================================
# def detect_decelerations(signal, baseline, sampling_rate):
#     min_drop = 15
#     min_duration = int(15 * sampling_rate)
#     decel_count = 0
#     i = 0
#     N = len(signal)
#     while i < N:
#         if signal[i] < baseline - min_drop:
#             start = i
#             while i < N and signal[i] < baseline - 5:
#                 i += 1
#             if i - start >= min_duration:
#                 decel_count += 1
#         else:
#             i += 1
#     return decel_count

# def detect_accelerations(signal, baseline, sampling_rate):
#     min_rise = 15
#     min_duration = int(15 * sampling_rate)
#     accels = 0
#     i = 0
#     N = len(signal)
#     while i < N:
#         if signal[i] > baseline + min_rise:
#             start = i
#             while i < N and signal[i] > baseline + 5:
#                 i += 1
#             if i - start >= min_duration:
#                 accels += 1
#         else:
#             i += 1
#     return accels

# def compute_features(fhr_signal, uc_signal, sampling_rate):
#     window = int(60 * sampling_rate)
#     baselines = [np.mean(fhr_signal[i:i+window]) for i in range(0, len(fhr_signal), window)
#                  if i+window <= len(fhr_signal)]
#     baseline = float(np.mean(baselines))

#     total_sec = len(fhr_signal) / sampling_rate
#     total_min = total_sec / 60
#     factor_10 = total_min / 10 if total_min > 0 else 1

#     accel_count = detect_accelerations(fhr_signal, baseline, sampling_rate)
#     decels = detect_decelerations(fhr_signal, baseline, sampling_rate)

#     fetal_movement = int(np.sum(np.diff(uc_signal) > 10) / factor_10)
#     uc_peaks, _ = find_peaks(uc_signal, height=np.nanmean(uc_signal) + 10)
#     uterine_contractions = int(len(uc_peaks) / factor_10)

#     return {
#         "baseline value": baseline,
#         "accelerations": int(accel_count),
#         "fetal_movement": int(fetal_movement),
#         "uterine_contractions": int(uterine_contractions),
#         "prolongued_decelerations": int(decels)
#     }

# # =======================================================
# # ROUTES
# # =======================================================
# @app.post("/predict/")
# async def predict_ctg(file: UploadFile = File(...)):

#     contents = await file.read()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         tmp.write(contents)
#         path = tmp.name

#     try:
#         # -----------------------------------------
#         # STEP 1 — CNN Binary Classifier FIRST
#         # -----------------------------------------
#         binary_label, binary_score = classify_ctg_or_nonctg(path)

#         if binary_label == "Non-CTG":
#             # Upload to Cloudinary
#             upload_result = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
#             img_url = upload_result.get("secure_url")

#             record = {
#                 "timestamp": datetime.utcnow() + timedelta(hours=6),
#                 "isCTG": False,
#                 "binaryConfidence": binary_score,
#                 "imageUrl": img_url
#             }
#             ctg_collection.insert_one(record)

#             return {
#                 "isCTG": False,
#                 "message": "The uploaded image is NOT a CTG graph.",
#                 "confidence": binary_score,
#                 "imageUrl": img_url
#             }

#         # -----------------------------------------
#         # STEP 2 — Full CTG Extraction & Analysis
#         # -----------------------------------------
#         fhr_signal, uc_signal, fs = extract_ctg_signals(path)
#         features = compute_features(fhr_signal, uc_signal, fs)

#         df = pd.DataFrame([features])
#         clinical_label = model_ctg_class.predict(df)
#         client_label = convert_to_client_label(clinical_label)

#         upload_result = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
#         img_url = upload_result.get("secure_url")

#         record = {
#             "timestamp": datetime.utcnow() + timedelta(hours=6),
#             "isCTG": True,
#             "binaryConfidence": binary_score,
#             "ctgDetected": clinical_label,
#             "clientLabel": client_label,
#             "features": features,
#             "imageUrl": img_url
#         }

#         result = ctg_collection.insert_one(record)

#         return {
#             "isCTG": True,
#             "label": client_label,
#             "clinical_label": clinical_label,
#             "features": features,
#             "confidence": binary_score,
#             "imageUrl": img_url,
#             "record_id": str(result.inserted_id)
#         }

#     finally:
#         os.remove(path)

# # =======================================================
# @app.get("/records")
# def records():
#     recs = list(ctg_collection.find().sort("timestamp", -1))
#     return {"records": [
#         {
#             "id": str(r["_id"]),
#             "timestamp": r.get("timestamp"),
#             "isCTG": r.get("isCTG"),
#             "binaryConfidence": r.get("binaryConfidence"),
#             "ctgDetected": r.get("ctgDetected"),
#             "clientLabel": r.get("clientLabel"),
#             "features": r.get("features"),
#             "imageUrl": r.get("imageUrl")
#         }
#         for r in recs
#     ]}

# # =======================================================
# @app.delete("/records/{record_id}")
# def delete_record(record_id: str):
#     r = ctg_collection.delete_one({"_id": ObjectId(record_id)})
#     if r.deleted_count == 0:
#         raise HTTPException(status_code=404, detail="Record not found")
#     return {"detail": "Record deleted"}

# # =======================================================
# @app.get("/api/analysis")
# def get_analysis():
#     records = list(ctg_collection.find({}, {"_id": 0}))
#     if not records:
#         return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

#     df = pd.DataFrame(records)

#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

#     pivot = df.pivot_table(
#         index="date",
#         columns="ctgDetected",
#         aggfunc="size",
#         fill_value=0
#     ).reset_index()

#     pivot = pivot.rename(columns={
#         "Normal": "N",
#         "Suspect": "S",
#         "Pathologic": "P"
#     })

#     time_series = pivot.to_dict(orient="records")

#     counts = Counter(df["ctgDetected"])

#     nspStats = {
#         "Normal": int(counts.get("Normal", 0)),
#         "Suspect": int(counts.get("Suspect", 0)),
#         "Pathologic": int(counts.get("Pathologic", 0)),
#     }

#     return {
#         "predictions": time_series,
#         "nspStats": nspStats
#     }

# # =======================================================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# server.py – FINAL VERSION (CTG 5-Feature Model Integrated)

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
import tensorflow as tf
import math
import logging

# ------------------------------------------------------------
# ENV + LOGGING
# ------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# CLIENT-FACING LABELS
# ------------------------------------------------------------
def convert_to_client_label(num):
    return {
        1: "Normal",
        2: "Suspicious",
        3: "Pathological"
    }.get(num, "Unknown")



# ------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------
app = FastAPI(title="Druk Health CTG – 5-Feature Model")

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

# ------------------------------------------------------------
# LOAD 5-FEATURE CLASSIFIER
# ------------------------------------------------------------
MODEL_5F_PATH = "ctg_5feature_model.pkl"

if not os.path.exists(MODEL_5F_PATH):
    raise FileNotFoundError("Missing model file: ctg_5feature_model.pkl")

model_5f = joblib.load(MODEL_5F_PATH)
logging.info("✅ Loaded CTG 5-feature model with DecisionTreeClassifier")


# ------------------------------------------------------------
# LOAD BINARY CTG vs NON-CTG CNN
# ------------------------------------------------------------
CNN_PATH = "CTG_vs_NonCTG(1).keras"
binary_ctg_model = tf.keras.models.load_model(CNN_PATH)
logging.info("✅ Loaded CTG vs Non-CTG CNN")


def classify_ctg_or_nonctg(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    score = float(binary_ctg_model.predict(arr)[0][0])
    return ("CTG" if score > 0.5 else "Non-CTG"), score


# ======================================================================
# ========================== IMAGE → SIGNAL =============================
# ======================================================================

def detect_grid(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 7)
    h, w = th.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//15, 1))
    hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)
    rows = np.where(np.sum(hl, axis=1) > np.percentile(np.sum(hl,axis=1), 95))[0]
    if len(rows) < 2: return None
    d = np.diff(rows)
    d = d[d > 3]
    if len(d)==0: return None
    return int(np.median(d) * 5)


def extract_trace(roi):
    blur = cv2.GaussianBlur(roi, (3,3), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,15,5)

    ys=[]
    h,w=th.shape
    for x in range(w):
        idx=np.where(th[:,x]>0)[0]
        ys.append(np.median(idx) if len(idx)>0 else np.nan)

    ys=np.array(ys)
    xs=np.arange(len(ys))
    ys[np.isnan(ys)] = np.interp(xs[np.isnan(ys)], xs[~np.isnan(ys)], ys[~np.isnan(ys)])
    return ys


def extract_ctg_signals(path):
    img=cv2.imread(path)
    h,w=img.shape[:2]
    if h>w:
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h,w=img.shape[:2]

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)

    px_large = detect_grid(gray)
    if px_large is None:
        px_large=180  # fallback

    top=int(h*0.05)
    mid=int(h*0.45)
    bottom=int(h*0.95)

    fhr_y = extract_trace(gray[top:mid,:]) + top
    uc_y = extract_trace(gray[mid:bottom,:])

    bpm_pp = 30/px_large
    ref_y=(top+mid)/2
    fhr_bpm = 140 - (fhr_y - ref_y)*bpm_pp
    fhr_bpm=medfilt(fhr_bpm,7)
    fhr_bpm=np.clip(fhr_bpm,40,220)

    sec_per_px = 60/px_large
    fs=1/sec_per_px

    uc_norm = (uc_y - np.min(uc_y)) / (np.ptp(uc_y)+1e-6)
    return fhr_bpm, uc_norm, fs


# ======================================================================
# =================== FEATURE COMPUTATION (5 FEATURES) ==================
# ======================================================================

def compute_features_5f(fhr, uc, fs):
    samples_per_min = max(1, int(fs * 60))

    # ---- Baseline: first 1 minute ----
    baseline = float(np.median(fhr[:samples_per_min]))

    # ---- RAW variability (std) ----
    std_val = float(np.std(fhr[:samples_per_min]))

    # ---- Convert STD → Variability Category ----
    if std_val < 5:
        variability = 0         # absent
    elif std_val < 10:
        variability = 1         # reduced
    elif std_val < 25:
        variability = 2         # moderate
    else:
        variability = 3         # increased

    # ---- First 20 minutes ----
    samples_20 = samples_per_min * 20
    fwin = fhr[:samples_20]

    def detect(kind):
        amp = 15
        dur = int(fs * 15)
        N = len(fwin)
        i = 0
        c = 0
        while i < N:
            if kind == "accel" and fwin[i] > baseline + amp:
                s = i
                while i < N and fwin[i] > baseline + 5:
                    i += 1
                if i - s >= dur:
                    c += 1
            elif kind == "decel" and fwin[i] < baseline - amp:
                s = i
                while i < N and fwin[i] < baseline - 5:
                    i += 1
                if i - s >= dur:
                    c += 1
            else:
                i += 1
        return c

    acceleration = detect("accel")
    deceleration = detect("decel")

    # ---- Uterine contractions using peaks ----
    peaks, _ = find_peaks(uc, height=np.mean(uc) + 0.2)
    uterine = len(peaks)

    return {
        "acceleration": acceleration,
        "deceleration": deceleration,
        "baseline": float(round(baseline, 2)),
        "uterine_contraction": uterine,
        "variability": variability
    }


# ======================================================================
# ============================== API ===================================
# ======================================================================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as tmp:
        tmp.write(contents)
        path=tmp.name

    try:
        # --- Step 1: Binary CTG Check ---
        binary_label, score = classify_ctg_or_nonctg(path)

        if binary_label=="Non-CTG":
            up=cloudinary.uploader.upload(path,folder="drukhealth_ctg")
            return {"isCTG":False,"message":"Not a CTG graph","confidence":score,"imageUrl":up["secure_url"]}

        # --- Step 2: Extract CTG signals ---
        fhr, uc, fs = extract_ctg_signals(path)

        # --- Step 3: Compute 5 features ---
        features = compute_features_5f(fhr, uc, fs)

        # --- Step 4: Prediction ---
        df=pd.DataFrame([features])
        pred_num=int(model_5f.predict(df)[0])
        label=convert_to_client_label(pred_num)

        # --- Step 5: Upload & Save ---
        up=cloudinary.uploader.upload(path,folder="drukhealth_ctg")

        record={
            "timestamp": datetime.utcnow()+timedelta(hours=6),
            "isCTG": True,
            "binaryConfidence": score,
            "classNumeric": pred_num,
            "labelClient": label,
            "features": features,
            "imageUrl": up["secure_url"]
        }
        res=ctg_collection.insert_one(record)

        return {
            "isCTG":True,
            "label":label,
            "numeric":pred_num,
            "features":features,
            "confidence":score,
            "imageUrl":up["secure_url"],
            "record_id":str(res.inserted_id)
        }

    finally:
        try: os.remove(path)
        except: pass


@app.get("/records")
def records():
    recs=list(ctg_collection.find().sort("timestamp",-1))
    return {"records":[
        {
            "id":str(r["_id"]),
            "timestamp":r.get("timestamp"),
            "isCTG":r.get("isCTG"),
            "binaryConfidence":r.get("binaryConfidence"),
            "labelClient":r.get("labelClient"),
            "features":r.get("features"),
            "imageUrl":r.get("imageUrl")
        } for r in recs
    ]}


@app.delete("/records/{record_id}")
def delete_record(record_id):
    r=ctg_collection.delete_one({"_id":ObjectId(record_id)})
    if r.deleted_count==0:
        raise HTTPException(404,"Record not found")
    return {"detail":"Record deleted"}


@app.get("/api/analysis")
def analysis():
    data=list(ctg_collection.find({},{"_id":0}))
    if not data: return {"predictions":[], "nspStats":{"Normal":0,"Suspect":0,"Pathologic":0}}

    df=pd.DataFrame(data)
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    df["date"]=df["timestamp"].dt.strftime("%Y-%m-%d")

    pivot=df.pivot_table(index="date",columns="classNumeric",aggfunc="size",fill_value=0)
    counts=df["classNumeric"].value_counts()

    return {
        "predictions": pivot.reset_index().to_dict(orient="records"),
        "nspStats": {
            "Normal": int(counts.get(1,0)),
            "Suspect": int(counts.get(2,0)),
            "Pathologic": int(counts.get(3,0))
        }
    }


# RUN
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
