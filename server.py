# # ==========================================
# # server.py — Druk Health CTG AI Backend (FINAL - Using drukhealth.ctgscans + Cloudinary)
# # ==========================================

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
# from scipy.signal import find_peaks
# from collections import Counter
# import cloudinary
# import cloudinary.uploader
# from dotenv import load_dotenv

# # ------------------------------
# # Load environment (optional)
# # ------------------------------
# load_dotenv()

# # ------------------------------
# # Initialize FastAPI app
# # ------------------------------
# app = FastAPI(title="Druk Health CTG AI Backend")

# origins = [
#     "http://localhost:5173",
#     "https://drukhealthfe.vercel.app",
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
# # Cloudinary Config
# # ------------------------------
# CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b")
# CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "522272821951884")
# CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")

# cloudinary.config(
#     cloud_name=CLOUDINARY_CLOUD_NAME,
#     api_key=CLOUDINARY_API_KEY,
#     api_secret=CLOUDINARY_API_SECRET
# )

# # ------------------------------
# # Load trained model
# # ------------------------------
# model_ctg_class = joblib.load("decision_tree_all_cardio_features.pkl")
# print("✅ Loaded Decision Tree model with features:\n", model_ctg_class.feature_names_in_)

# # =======================================================
# # HELPER FUNCTIONS
# # =======================================================
# def extract_ctg_signals(image_path, fhr_top_ratio=0.55, bpm_per_cm=30,
#                         toco_per_cm=25, paper_speed_cm_min=2, fhr_min_line=50):
#     """Extract FHR and UC signals from CTG image using OpenCV."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if np.mean(gray) > 127:
#         gray = cv2.bitwise_not(gray)

#     height, width = gray.shape
#     fhr_img = gray[0:int(fhr_top_ratio * height), :]
#     uc_img = gray[int(fhr_top_ratio * height):, :]

#     def extract_signal(trace_img):
#         _, thresh = cv2.threshold(trace_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         h, w = thresh.shape
#         signal = []
#         for x in range(w):
#             y_pixels = np.where(thresh[:, x] > 0)[0]
#             y = np.median(y_pixels) if len(y_pixels) > 0 else np.nan
#             signal.append(y)
#         signal = pd.Series(signal).interpolate(limit_direction="both").values
#         return h - signal

#     fhr_signal = extract_signal(fhr_img)
#     uc_signal = extract_signal(uc_img)

#     px_per_cm = height / 10.0
#     bpm_per_px = bpm_per_cm / px_per_cm
#     toco_per_px = toco_per_cm / px_per_cm
#     fhr_signal = fhr_min_line + fhr_signal * bpm_per_px
#     uc_signal = uc_signal * toco_per_px
#     px_per_sec = (paper_speed_cm_min / 60.0) * px_per_cm
#     time_axis = np.arange(len(fhr_signal)) / px_per_sec

#     return fhr_signal, uc_signal, time_axis


# def compute_model_features(fhr_signal, uc_signal, time_axis):
#     """Compute SisPorto-style features for CTG classification."""
#     features = {}
#     duration = time_axis[-1] - time_axis[0]
#     baseline = np.mean(fhr_signal)
#     fhr_diff = np.diff(fhr_signal)

#     # --- Accelerations ---
#     accel_count = 0
#     in_accel = False
#     start_idx = None
#     for i in range(len(fhr_signal)):
#         if fhr_signal[i] > baseline + 5:
#             if not in_accel:
#                 in_accel = True
#                 start_idx = i
#         else:
#             if in_accel:
#                 dur = time_axis[i - 1] - time_axis[start_idx]
#                 amp = np.max(fhr_signal[start_idx:i]) - baseline
#                 if dur >= 5 and amp >= 5:
#                     accel_count += 1
#                 in_accel = False
#     features["Accelerations (SisPorto)"] = round(accel_count / (duration / 600.0), 2)

#     # --- Uterine contractions & fetal movements ---
#     uc_peaks, _ = find_peaks(uc_signal, height=15, distance=int(30 * (len(time_axis) / duration)))
#     fm_events, _ = find_peaks(np.diff(uc_signal), height=10, distance=int(5 * (len(time_axis) / duration)))
#     features["Uterine contractions (SisPorto)"] = round(len(uc_peaks) / (duration / 600.0), 2)
#     features["Fetal movements (SisPorto)"] = round(len(fm_events) / (duration / 600.0), 2)

#     # --- Decelerations ---
#     dips, _ = find_peaks(-fhr_signal, height=-(baseline - 15), distance=int(15))
#     severe, _ = find_peaks(-fhr_signal, height=-(baseline - 25), distance=int(15))
#     prolonged = [i for i in range(1, len(dips)) if (dips[i] - dips[i - 1]) > 120]
#     repetitive = np.sum(np.diff(dips) < 60)
#     features["Light decelerations (raw)"] = round(len(dips) / (duration / 600.0), 2)
#     features["Severe decelerations (raw)"] = round(len(severe) / (duration / 600.0), 2)
#     features["Prolonged decelerations (raw)"] = round(len(prolonged) / (duration / 600.0), 2)
#     features["Repetitive decelerations (raw)"] = round(repetitive / (duration / 600.0), 2)

#     # --- Baseline & variability ---
#     features["Baseline value (SisPorto)"] = round(float(baseline), 2)
#     features["Percentage time with abnormal short-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_diff) > 25) / len(fhr_diff), 2
#     )
#     features["Mean value of short-term variability (SisPorto)"] = round(np.mean(np.abs(fhr_diff)), 2)
#     features["Percentage time with abnormal long-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_signal - baseline) > 20) / len(fhr_signal), 2
#     )
#     features["Mean value of long-term variability (SisPorto)"] = round(np.std(fhr_signal), 2)

#     # --- Histogram ---
#     hist, bins = np.histogram(fhr_signal, bins=10)
#     features["Histogram width"] = round(bins[-1] - bins[0], 2)
#     features["Histogram minimum frequency"] = round(np.min(fhr_signal), 2)
#     features["Histogram maximum frequency"] = round(np.max(fhr_signal), 2)
#     features["Number of histogram peaks"] = int(np.max(hist))
#     features["Number of histogram zeros"] = int(np.sum(hist == 0))
#     features["Histogram mode"] = round(bins[np.argmax(hist)], 2)
#     features["Histogram mean"] = round(np.mean(fhr_signal), 2)
#     features["Histogram median"] = round(np.median(fhr_signal), 2)
#     features["Histogram variance"] = round(np.var(fhr_signal), 2)
#     features["Histogram tendency (-1=left asymmetric; 0=symmetric; 1=right asymmetric)"] = round(
#         fhr_signal[-1] - fhr_signal[0], 2
#     )

#     for col in model_ctg_class.feature_names_in_:
#         if col not in features:
#             features[col] = 0

#     return features


# # =======================================================
# # ROUTES
# # =======================================================
# @app.get("/")
# def home():
#     return {"message": "CTG AI Prediction API is running 🚀"}

# # -------------------------------------------------------
# # PREDICT & SAVE (NOW WITH IMAGE UPLOAD)
# # -------------------------------------------------------
# @app.post("/predict/")
# async def predict_ctg(file: UploadFile = File(...)):
#     """Upload CTG image → extract features → predict → upload to Cloudinary → store in MongoDB"""
#     contents = await file.read()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         tmp.write(contents)
#         tmp_path = tmp.name

#     try:
#         # --- Extract + Predict ---
#         fhr, uc, t = extract_ctg_signals(tmp_path)
#         features = compute_model_features(fhr, uc, t)
#         df = pd.DataFrame([features])[model_ctg_class.feature_names_in_]
#         pred = model_ctg_class.predict(df)[0]
#         label = {1: "Normal", 2: "Suspect", 3: "Pathologic"}.get(pred, "Unknown")

#         # --- Upload to Cloudinary ---
#         upload_result = cloudinary.uploader.upload(tmp_path, folder="drukhealth_ctg")
#         image_url = upload_result.get("secure_url")

#         # --- Store in MongoDB ---
#         record = {
#             "timestamp": datetime.utcnow() + timedelta(hours=6),
#             "ctgDetected": label,
#             "features": features,
#             "imageUrl": image_url
#         }
#         result = ctg_collection.insert_one(record)
#         print(f"✅ Saved record {result.inserted_id} ({label})")

#         return {
#             "prediction": int(pred),
#             "label": label,
#             "features": features,
#             "imageUrl": image_url,
#             "record_id": str(result.inserted_id),
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # -------------------------------------------------------
# # GET ALL RECORDS
# # -------------------------------------------------------
# @app.get("/records")
# def get_records():
#     """Return all stored CTG scan records (with image + features only)."""
#     try:
#         records = list(ctg_collection.find().sort("timestamp", -1))
#         formatted = []
#         for rec in records:
#             if rec.get("features") and rec.get("imageUrl"):
#                 formatted.append({
#                     "id": str(rec["_id"]),
#                     "timestamp": rec.get("timestamp"),
#                     "ctgDetected": rec.get("ctgDetected", "Unknown"),
#                     "features": rec.get("features", {}),
#                     "imageUrl": rec.get("imageUrl", "")
#                 })
#         return {"records": formatted}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------------------------------
# # DELETE RECORD
# # -------------------------------------------------------
# @app.delete("/records/{record_id}")
# def delete_record(record_id: str):
#     try:
#         result = ctg_collection.delete_one({"_id": ObjectId(record_id)})
#         if result.deleted_count == 0:
#             raise HTTPException(status_code=404, detail="Record not found")
#         return {"detail": "Record deleted successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------------------------------
# # DASHBOARD ANALYSIS
# # -------------------------------------------------------
# @app.get("/api/analysis")
# def get_analysis():
#     """Return summary stats for dashboard charts."""
#     records = list(ctg_collection.find({}, {"_id": 0}))
#     if not records:
#         return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

#     df = pd.DataFrame(records)
#     df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
#     pivot = df.pivot_table(index="date", columns="ctgDetected", aggfunc="size", fill_value=0).reset_index()
#     pivot = pivot.rename_axis(None, axis=1)
#     pivot = pivot.rename(columns={"Normal": "N", "Suspect": "S", "Pathologic": "P"})
#     time_series = pivot.to_dict(orient="records")

#     counts = Counter(df["ctgDetected"])
#     nspStats = {
#         "Normal": int(counts.get("Normal", 0)),
#         "Suspect": int(counts.get("Suspect", 0)),
#         "Pathologic": int(counts.get("Pathologic", 0)),
#     }

#     return {"predictions": time_series, "nspStats": nspStats}


# # =======================================================
# # Run server
# # =======================================================
# if __name__ == "__main__":
#     import uvicorn
#     import os
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)














# NYCKEL_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwczovL3d3dy5ueWNrZWwuY29tIiwibmJmIjoxNzYyODc5MzMxLCJpYXQiOjE3NjI4NzkzMzEsImV4cCI6MTc2Mjg4MjkzMSwic2NvcGUiOlsiYXBpIl0sImNsaWVudF9pZCI6Imh5Njl3dHN4ZG9vczkwbzg1NDJua3FmNTg2eXN2b2ZsIiwianRpIjoiQkRFNjJCMDhEOUUzNkUwNDgyNzYwQTdFQTdGMDQ1NjIifQ.hY7znBCxWQ-DSKTa6Ho8k6Tol6J1fiRrzR5bh-j8naTg_ltdm2Gg1PZthrI5PKTgmfWBXIeZndumXi4E8pNwQgVB595BJ9vDqTIsJb-y-yVVnOshq1JZa863HMeg0cn3emr0jpeAO6u5x9WLgdevAJmZDpdYh1qLNMKruZ2aD6MnVosM39o5ioGLucNqtzM4vqGiiHWiXVgZ5A-NWBGOTD8X1Kg1Y0hXv7GYakVtFC43uh90ptuk8FUsCA4BmJiZ14BDN9V_F-SPHsLO4afte10anxJFhEsEeoMvBB3j2U8COkTKGwnvnU3QA_DQCVYx9zoy3Q10JJQlkvmg2o9u_w"

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import pandas as pd
# import numpy as np
# import cv2
# import tempfile
# import os
# import requests
# from PIL import Image
# import joblib
# from scipy.signal import find_peaks

# # --- FastAPI app ---
# app = FastAPI(title="Druk Health CTG AI Backend")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Load local model ---
# model = joblib.load("decision_tree_all_cardio_features.pkl")

# # --- Nyckel API config ---
# NYCKEL_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwczovL3d3dy5ueWNrZWwuY29tIiwibmJmIjoxNzYyODc5MzMxLCJpYXQiOjE3NjI4NzkzMzEsImV4cCI6MTc2Mjg4MjkzMSwic2NvcGUiOlsiYXBpIl0sImNsaWVudF9pZCI6Imh5Njl3dHN4ZG9vczkwbzg1NDJua3FmNTg2eXN2b2ZsIiwianRpIjoiQkRFNjJCMDhEOUUzNkUwNDgyNzYwQTdFQTdGMDQ1NjIifQ.hY7znBCxWQ-DSKTa6Ho8k6Tol6J1fiRrzR5bh-j8naTg_ltdm2Gg1PZthrI5PKTgmfWBXIeZndumXi4E8pNwQgVB595BJ9vDqTIsJb-y-yVVnOshq1JZa863HMeg0cn3emr0jpeAO6u5x9WLgdevAJmZDpdYh1qLNMKruZ2aD6MnVosM39o5ioGLucNqtzM4vqGiiHWiXVgZ5A-NWBGOTD8X1Kg1Y0hXv7GYakVtFC43uh90ptuk8FUsCA4BmJiZ14BDN9V_F-SPHsLO4afte10anxJFhEsEeoMvBB3j2U8COkTKGwnvnU3QA_DQCVYx9zoy3Q10JJQlkvmg2o9u_w"
# NYCKEL_FUNCTION_ID = "cdk3y4u8ff799uh3"
# NYCKEL_URL = f"https://www.nyckel.com/v1/functions/{NYCKEL_FUNCTION_ID}/invoke"

# # --- Nyckel API call ---
# def classify_with_nyckel(image_path):
#     try:
#         tmp_path = image_path
#         if image_path.lower().endswith(".png"):
#             tmp_path = image_path.replace(".png", ".jpg")
#             with Image.open(image_path) as im:
#                 im.convert("RGB").save(tmp_path, format="JPEG")

#         with open(tmp_path, "rb") as f:
#             files = {"data": ("image.jpg", f, "image/jpeg")}
#             headers = {"Authorization": f"Bearer {NYCKEL_KEY}"}
#             response = requests.post(NYCKEL_URL, headers=headers, files=files)
#             response.raise_for_status()
#             return response.json()
#     except Exception as e:
#         return {"error": str(e)}
#     finally:
#         if tmp_path != image_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # =======================================================
# # HELPER FUNCTIONS
# # =======================================================
# def extract_ctg_signals(image_path, fhr_top_ratio=0.55, bpm_per_cm=30,
#                         toco_per_cm=25, paper_speed_cm_min=2, fhr_min_line=50):
#     """Extract FHR and UC signals from CTG image using OpenCV."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if np.mean(gray) > 127:
#         gray = cv2.bitwise_not(gray)

#     height, width = gray.shape
#     fhr_img = gray[0:int(fhr_top_ratio * height), :]
#     uc_img = gray[int(fhr_top_ratio * height):, :]

#     def extract_signal(trace_img):
#         _, thresh = cv2.threshold(trace_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         h, w = thresh.shape
#         signal = []
#         for x in range(w):
#             y_pixels = np.where(thresh[:, x] > 0)[0]
#             y = np.median(y_pixels) if len(y_pixels) > 0 else np.nan
#             signal.append(y)
#         signal = pd.Series(signal).interpolate(limit_direction="both").values
#         return h - signal

#     fhr_signal = extract_signal(fhr_img)
#     uc_signal = extract_signal(uc_img)

#     px_per_cm = height / 10.0
#     bpm_per_px = bpm_per_cm / px_per_cm
#     toco_per_px = toco_per_cm / px_per_cm
#     fhr_signal = fhr_min_line + fhr_signal * bpm_per_px
#     uc_signal = uc_signal * toco_per_px
#     px_per_sec = (paper_speed_cm_min / 60.0) * px_per_cm
#     time_axis = np.arange(len(fhr_signal)) / px_per_sec

#     return fhr_signal, uc_signal, time_axis


# def compute_model_features(fhr_signal, uc_signal, time_axis):
#     """Compute SisPorto-style features for CTG classification."""
#     features = {}
#     duration = time_axis[-1] - time_axis[0]
#     baseline = np.mean(fhr_signal)
#     fhr_diff = np.diff(fhr_signal)

#     # --- Accelerations ---
#     accel_count = 0
#     in_accel = False
#     start_idx = None
#     for i in range(len(fhr_signal)):
#         if fhr_signal[i] > baseline + 5:
#             if not in_accel:
#                 in_accel = True
#                 start_idx = i
#         else:
#             if in_accel:
#                 dur = time_axis[i - 1] - time_axis[start_idx]
#                 amp = np.max(fhr_signal[start_idx:i]) - baseline
#                 if dur >= 5 and amp >= 5:
#                     accel_count += 1
#                 in_accel = False
#     features["Accelerations (SisPorto)"] = round(accel_count / (duration / 600.0), 2)

#     # --- Uterine contractions & fetal movements ---
#     uc_peaks, _ = find_peaks(uc_signal, height=15, distance=int(30 * (len(time_axis) / duration)))
#     fm_events, _ = find_peaks(np.diff(uc_signal), height=10, distance=int(5 * (len(time_axis) / duration)))
#     features["Uterine contractions (SisPorto)"] = round(len(uc_peaks) / (duration / 600.0), 2)
#     features["Fetal movements (SisPorto)"] = round(len(fm_events) / (duration / 600.0), 2)

#     # --- Decelerations ---
#     dips, _ = find_peaks(-fhr_signal, height=-(baseline - 15), distance=int(15))
#     severe, _ = find_peaks(-fhr_signal, height=-(baseline - 25), distance=int(15))
#     prolonged = [i for i in range(1, len(dips)) if (dips[i] - dips[i - 1]) > 120]
#     repetitive = np.sum(np.diff(dips) < 60)
#     features["Light decelerations (raw)"] = round(len(dips) / (duration / 600.0), 2)
#     features["Severe decelerations (raw)"] = round(len(severe) / (duration / 600.0), 2)
#     features["Prolonged decelerations (raw)"] = round(len(prolonged) / (duration / 600.0), 2)
#     features["Repetitive decelerations (raw)"] = round(repetitive / (duration / 600.0), 2)

#     # --- Baseline & variability ---
#     features["Baseline value (SisPorto)"] = round(float(baseline), 2)
#     features["Percentage time with abnormal short-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_diff) > 25) / len(fhr_diff), 2
#     )
#     features["Mean value of short-term variability (SisPorto)"] = round(np.mean(np.abs(fhr_diff)), 2)
#     features["Percentage time with abnormal long-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_signal - baseline) > 20) / len(fhr_signal), 2
#     )
#     features["Mean value of long-term variability (SisPorto)"] = round(np.std(fhr_signal), 2)

#     # --- Histogram ---
#     hist, bins = np.histogram(fhr_signal, bins=10)
#     features["Histogram width"] = round(bins[-1] - bins[0], 2)
#     features["Histogram minimum frequency"] = round(np.min(fhr_signal), 2)
#     features["Histogram maximum frequency"] = round(np.max(fhr_signal), 2)
#     features["Number of histogram peaks"] = int(np.max(hist))
#     features["Number of histogram zeros"] = int(np.sum(hist == 0))
#     features["Histogram mode"] = round(bins[np.argmax(hist)], 2)
#     features["Histogram mean"] = round(np.mean(fhr_signal), 2)
#     features["Histogram median"] = round(np.median(fhr_signal), 2)
#     features["Histogram variance"] = round(np.var(fhr_signal), 2)
#     features["Histogram tendency (-1=left asymmetric; 0=symmetric; 1=right asymmetric)"] = round(
#         fhr_signal[-1] - fhr_signal[0], 2
#     )

#     for col in model_ctg_class.feature_names_in_:
#         if col not in features:
#             features[col] = 0

#     return features


# # --- Prediction endpoint ---
# @app.post("/predict/")
# async def predict_ctg(file: UploadFile = File(...)):
#     contents = await file.read()
#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             tmp.write(contents)
#             tmp_path = tmp.name

#         # Local model prediction
#         fhr, uc, t, _ = extract_ctg_signals(tmp_path)
#         features = compute_model_features(fhr, uc, t)
#         df = pd.DataFrame([features])
#         for col in model.feature_names_in_:
#             if col not in df.columns:
#                 df[col] = 0
#         df = df[model.feature_names_in_]
#         local_pred = model.predict(df)[0]
#         local_label = {1: "Normal", 2: "Suspect", 3: "Pathologic"}[local_pred]

#         # Nyckel prediction
#         nyckel_result = classify_with_nyckel(tmp_path)

#         return {
#             "local_prediction": {"prediction": int(local_pred), "label": local_label},
#             "nyckel_prediction": nyckel_result,
#             "features": features
#         }

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# ==========================================
# server.py — Druk Health CTG AI Backend (FINAL - Using drukhealth.ctgscans + Cloudinary)
# ==========================================

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
# from scipy.signal import find_peaks
# from collections import Counter
# import cloudinary
# import cloudinary.uploader
# from dotenv import load_dotenv

# # ------------------------------
# # Load environment (optional)
# # ------------------------------
# load_dotenv()

# # --- Nyckel API config ---
# NYCKEL_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwczovL3d3dy5ueWNrZWwuY29tIiwibmJmIjoxNzYyODc5MzMxLCJpYXQiOjE3NjI4NzkzMzEsImV4cCI6MTc2Mjg4MjkzMSwic2NvcGUiOlsiYXBpIl0sImNsaWVudF9pZCI6Imh5Njl3dHN4ZG9vczkwbzg1NDJua3FmNTg2eXN2b2ZsIiwianRpIjoiQkRFNjJCMDhEOUUzNkUwNDgyNzYwQTdFQTdGMDQ1NjIifQ.hY7znBCxWQ-DSKTa6Ho8k6Tol6J1fiRrzR5bh-j8naTg_ltdm2Gg1PZthrI5PKTgmfWBXIeZndumXi4E8pNwQgVB595BJ9vDqTIsJb-y-yVVnOshq1JZa863HMeg0cn3emr0jpeAO6u5x9WLgdevAJmZDpdYh1qLNMKruZ2aD6MnVosM39o5ioGLucNqtzM4vqGiiHWiXVgZ5A-NWBGOTD8X1Kg1Y0hXv7GYakVtFC43uh90ptuk8FUsCA4BmJiZ14BDN9V_F-SPHsLO4afte10anxJFhEsEeoMvBB3j2U8COkTKGwnvnU3QA_DQCVYx9zoy3Q10JJQlkvmg2o9u_w"
# NYCKEL_FUNCTION_ID = "cdk3y4u8ff799uh3"
# NYCKEL_URL = f"https://www.nyckel.com/v1/functions/{NYCKEL_FUNCTION_ID}/invoke"

# # --- Nyckel API call ---
# def classify_with_nyckel(image_path):
#     try:
#         tmp_path = image_path
#         if image_path.lower().endswith(".png"):
#             tmp_path = image_path.replace(".png", ".jpg")
#             with Image.open(image_path) as im:
#                 im.convert("RGB").save(tmp_path, format="JPEG")

#         with open(tmp_path, "rb") as f:
#             files = {"data": ("image.jpg", f, "image/jpeg")}
#             headers = {"Authorization": f"Bearer {NYCKEL_KEY}"}
#             response = requests.post(NYCKEL_URL, headers=headers, files=files)
#             response.raise_for_status()
#             return response.json()
#     except Exception as e:
#         return {"error": str(e)}
#     finally:
#         if tmp_path != image_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # ------------------------------
# # Initialize FastAPI app
# # ------------------------------
# app = FastAPI(title="Druk Health CTG AI Backend")

# origins = [
#     "http://localhost:5173",
#     "https://drukhealthfe.vercel.app",
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
# # Cloudinary Config
# # ------------------------------
# CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b")
# CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "522272821951884")
# CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")

# cloudinary.config(
#     cloud_name=CLOUDINARY_CLOUD_NAME,
#     api_key=CLOUDINARY_API_KEY,
#     api_secret=CLOUDINARY_API_SECRET
# )

# # ------------------------------
# # Load trained model
# # ------------------------------
# model_ctg_class = joblib.load("decision_tree_all_cardio_features.pkl")
# print("✅ Loaded Decision Tree model with features:\n", model_ctg_class.feature_names_in_)

# # =======================================================
# # HELPER FUNCTIONS
# # =======================================================
# def extract_ctg_signals(image_path, fhr_top_ratio=0.55, bpm_per_cm=30,
#                         toco_per_cm=25, paper_speed_cm_min=2, fhr_min_line=50):
#     """Extract FHR and UC signals from CTG image using OpenCV."""
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if np.mean(gray) > 127:
#         gray = cv2.bitwise_not(gray)

#     height, width = gray.shape
#     fhr_img = gray[0:int(fhr_top_ratio * height), :]
#     uc_img = gray[int(fhr_top_ratio * height):, :]

#     def extract_signal(trace_img):
#         _, thresh = cv2.threshold(trace_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         h, w = thresh.shape
#         signal = []
#         for x in range(w):
#             y_pixels = np.where(thresh[:, x] > 0)[0]
#             y = np.median(y_pixels) if len(y_pixels) > 0 else np.nan
#             signal.append(y)
#         signal = pd.Series(signal).interpolate(limit_direction="both").values
#         return h - signal

#     fhr_signal = extract_signal(fhr_img)
#     uc_signal = extract_signal(uc_img)

#     px_per_cm = height / 10.0
#     bpm_per_px = bpm_per_cm / px_per_cm
#     toco_per_px = toco_per_cm / px_per_cm
#     fhr_signal = fhr_min_line + fhr_signal * bpm_per_px
#     uc_signal = uc_signal * toco_per_px
#     px_per_sec = (paper_speed_cm_min / 60.0) * px_per_cm
#     time_axis = np.arange(len(fhr_signal)) / px_per_sec

#     return fhr_signal, uc_signal, time_axis


# def compute_model_features(fhr_signal, uc_signal, time_axis):
#     """Compute SisPorto-style features for CTG classification."""
#     features = {}
#     duration = time_axis[-1] - time_axis[0]
#     baseline = np.mean(fhr_signal)
#     fhr_diff = np.diff(fhr_signal)

#     # --- Accelerations ---
#     accel_count = 0
#     in_accel = False
#     start_idx = None
#     for i in range(len(fhr_signal)):
#         if fhr_signal[i] > baseline + 5:
#             if not in_accel:
#                 in_accel = True
#                 start_idx = i
#         else:
#             if in_accel:
#                 dur = time_axis[i - 1] - time_axis[start_idx]
#                 amp = np.max(fhr_signal[start_idx:i]) - baseline
#                 if dur >= 5 and amp >= 5:
#                     accel_count += 1
#                 in_accel = False
#     features["Accelerations (SisPorto)"] = round(accel_count / (duration / 600.0), 2)

#     # --- Uterine contractions & fetal movements ---
#     uc_peaks, _ = find_peaks(uc_signal, height=15, distance=int(30 * (len(time_axis) / duration)))
#     fm_events, _ = find_peaks(np.diff(uc_signal), height=10, distance=int(5 * (len(time_axis) / duration)))
#     features["Uterine contractions (SisPorto)"] = round(len(uc_peaks) / (duration / 600.0), 2)
#     features["Fetal movements (SisPorto)"] = round(len(fm_events) / (duration / 600.0), 2)

#     # --- Decelerations ---
#     dips, _ = find_peaks(-fhr_signal, height=-(baseline - 15), distance=int(15))
#     severe, _ = find_peaks(-fhr_signal, height=-(baseline - 25), distance=int(15))
#     prolonged = [i for i in range(1, len(dips)) if (dips[i] - dips[i - 1]) > 120]
#     repetitive = np.sum(np.diff(dips) < 60)
#     features["Light decelerations (raw)"] = round(len(dips) / (duration / 600.0), 2)
#     features["Severe decelerations (raw)"] = round(len(severe) / (duration / 600.0), 2)
#     features["Prolonged decelerations (raw)"] = round(len(prolonged) / (duration / 600.0), 2)
#     features["Repetitive decelerations (raw)"] = round(repetitive / (duration / 600.0), 2)

#     # --- Baseline & variability ---
#     features["Baseline value (SisPorto)"] = round(float(baseline), 2)
#     features["Percentage time with abnormal short-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_diff) > 25) / len(fhr_diff), 2
#     )
#     features["Mean value of short-term variability (SisPorto)"] = round(np.mean(np.abs(fhr_diff)), 2)
#     features["Percentage time with abnormal long-term variability (SisPorto)"] = round(
#         np.sum(np.abs(fhr_signal - baseline) > 20) / len(fhr_signal), 2
#     )
#     features["Mean value of long-term variability (SisPorto)"] = round(np.std(fhr_signal), 2)

#     # --- Histogram ---
#     hist, bins = np.histogram(fhr_signal, bins=10)
#     features["Histogram width"] = round(bins[-1] - bins[0], 2)
#     features["Histogram minimum frequency"] = round(np.min(fhr_signal), 2)
#     features["Histogram maximum frequency"] = round(np.max(fhr_signal), 2)
#     features["Number of histogram peaks"] = int(np.max(hist))
#     features["Number of histogram zeros"] = int(np.sum(hist == 0))
#     features["Histogram mode"] = round(bins[np.argmax(hist)], 2)
#     features["Histogram mean"] = round(np.mean(fhr_signal), 2)
#     features["Histogram median"] = round(np.median(fhr_signal), 2)
#     features["Histogram variance"] = round(np.var(fhr_signal), 2)
#     features["Histogram tendency (-1=left asymmetric; 0=symmetric; 1=right asymmetric)"] = round(
#         fhr_signal[-1] - fhr_signal[0], 2
#     )

#     for col in model_ctg_class.feature_names_in_:
#         if col not in features:
#             features[col] = 0

#     return features


# # =======================================================
# # ROUTES
# # =======================================================
# @app.get("/")
# def home():
#     return {"message": "CTG AI Prediction API is running 🚀"}

# # -------------------------------------------------------
# # PREDICT & SAVE (NOW WITH IMAGE UPLOAD)
# # -------------------------------------------------------
# @app.post("/predict/")
# async def predict_ctg(file: UploadFile = File(...)):
#     """Upload CTG image → extract features → predict → upload to Cloudinary → store in MongoDB"""
#     contents = await file.read()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         tmp.write(contents)
#         tmp_path = tmp.name

#     try:
#         # --- Extract + Predict ---
#         fhr, uc, t = extract_ctg_signals(tmp_path)
#         features = compute_model_features(fhr, uc, t)
#         df = pd.DataFrame([features])[model_ctg_class.feature_names_in_]
#         pred = model_ctg_class.predict(df)[0]
#         label = {1: "Normal", 2: "Suspect", 3: "Pathologic"}.get(pred, "Unknown")

#         # --- Upload to Cloudinary ---
#         upload_result = cloudinary.uploader.upload(tmp_path, folder="drukhealth_ctg")
#         image_url = upload_result.get("secure_url")

#         # --- Store in MongoDB ---
#         record = {
#             "timestamp": datetime.utcnow() + timedelta(hours=6),
#             "ctgDetected": label,
#             "features": features,
#             "imageUrl": image_url
#         }
#         result = ctg_collection.insert_one(record)
#         print(f"✅ Saved record {result.inserted_id} ({label})")

#         return {
#             "prediction": int(pred),
#             "label": label,
#             "features": features,
#             "imageUrl": image_url,
#             "record_id": str(result.inserted_id),
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # -------------------------------------------------------
# # GET ALL RECORDS
# # -------------------------------------------------------
# @app.get("/records")
# def get_records():
#     """Return all stored CTG scan records (with image + features only)."""
#     try:
#         records = list(ctg_collection.find().sort("timestamp", -1))
#         formatted = []
#         for rec in records:
#             if rec.get("features") and rec.get("imageUrl"):
#                 formatted.append({
#                     "id": str(rec["_id"]),
#                     "timestamp": rec.get("timestamp"),
#                     "ctgDetected": rec.get("ctgDetected", "Unknown"),
#                     "features": rec.get("features", {}),
#                     "imageUrl": rec.get("imageUrl", "")
#                 })
#         return {"records": formatted}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------------------------------
# # DELETE RECORD
# # -------------------------------------------------------
# @app.delete("/records/{record_id}")
# def delete_record(record_id: str):
#     try:
#         result = ctg_collection.delete_one({"_id": ObjectId(record_id)})
#         if result.deleted_count == 0:
#             raise HTTPException(status_code=404, detail="Record not found")
#         return {"detail": "Record deleted successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------------------------------
# # DASHBOARD ANALYSIS
# # -------------------------------------------------------
# @app.get("/api/analysis")
# def get_analysis():
#     """Return summary stats for dashboard charts."""
#     records = list(ctg_collection.find({}, {"_id": 0}))
#     if not records:
#         return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

#     df = pd.DataFrame(records)
#     df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
#     pivot = df.pivot_table(index="date", columns="ctgDetected", aggfunc="size", fill_value=0).reset_index()
#     pivot = pivot.rename_axis(None, axis=1)
#     pivot = pivot.rename(columns={"Normal": "N", "Suspect": "S", "Pathologic": "P"})
#     time_series = pivot.to_dict(orient="records")

#     counts = Counter(df["ctgDetected"])
#     nspStats = {
#         "Normal": int(counts.get("Normal", 0)),
#         "Suspect": int(counts.get("Suspect", 0)),
#         "Pathologic": int(counts.get("Pathologic", 0)),
#     }

#     return {"predictions": time_series, "nspStats": nspStats}


# # =======================================================
# # Run server
# # =======================================================
# if __name__ == "__main__":
#     import uvicorn
#     import os
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)



# ==========================================
# server.py — Druk Health CTG AI Backend (FINAL - Using drukhealth.ctgscans + Cloudinary + Nyckel)
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
from scipy.signal import find_peaks
from collections import Counter
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import requests  # <-- ⭐ ADDED for Nyckel API

# ------------------------------
# Load environment (optional)
# ------------------------------
load_dotenv()

# ------------------------------
# Initialize FastAPI app
# ------------------------------
app = FastAPI(title="Druk Health CTG AI Backend")

origins = [
    "http://localhost:5173",
    "https://drukhealthfe.vercel.app",
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
# Cloudinary Config
# ------------------------------
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "522272821951884")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# ------------------------------
# Nyckel API config  ⭐ ADDED
# ------------------------------
# # --- Nyckel API config ---
NYCKEL_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwczovL3d3dy5ueWNrZWwuY29tIiwibmJmIjoxNzYyODc5MzMxLCJpYXQiOjE3NjI4NzkzMzEsImV4cCI6MTc2Mjg4MjkzMSwic2NvcGUiOlsiYXBpIl0sImNsaWVudF9pZCI6Imh5Njl3dHN4ZG9vczkwbzg1NDJua3FmNTg2eXN2b2ZsIiwianRpIjoiQkRFNjJCMDhEOUUzNkUwNDgyNzYwQTdFQTdGMDQ1NjIifQ.hY7znBCxWQ-DSKTa6Ho8k6Tol6J1fiRrzR5bh-j8naTg_ltdm2Gg1PZthrI5PKTgmfWBXIeZndumXi4E8pNwQgVB595BJ9vDqTIsJb-y-yVVnOshq1JZa863HMeg0cn3emr0jpeAO6u5x9WLgdevAJmZDpdYh1qLNMKruZ2aD6MnVosM39o5ioGLucNqtzM4vqGiiHWiXVgZ5A-NWBGOTD8X1Kg1Y0hXv7GYakVtFC43uh90ptuk8FUsCA4BmJiZ14BDN9V_F-SPHsLO4afte10anxJFhEsEeoMvBB3j2U8COkTKGwnvnU3QA_DQCVYx9zoy3Q10JJQlkvmg2o9u_w"
NYCKEL_FUNCTION_ID = "cdk3y4u8ff799uh3"
NYCKEL_URL = f"https://www.nyckel.com/v1/functions/{NYCKEL_FUNCTION_ID}/invoke"

# ------------------------------
# Load trained model
# ------------------------------
model_ctg_class = joblib.load("decision_tree_all_cardio_features.pkl")
print("✅ Loaded Decision Tree model with features:\n", model_ctg_class.feature_names_in_)

# =======================================================
# HELPER FUNCTIONS (unchanged)
# =======================================================
def extract_ctg_signals(image_path, fhr_top_ratio=0.55, bpm_per_cm=30,
                        toco_per_cm=25, paper_speed_cm_min=2, fhr_min_line=50):
    """Extract FHR and UC signals from CTG image using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    height, width = gray.shape
    fhr_img = gray[0:int(fhr_top_ratio * height), :]
    uc_img = gray[int(fhr_top_ratio * height):, :]

    def extract_signal(trace_img):
        _, thresh = cv2.threshold(trace_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = thresh.shape
        signal = []
        for x in range(w):
            y_pixels = np.where(thresh[:, x] > 0)[0]
            y = np.median(y_pixels) if len(y_pixels) > 0 else np.nan
            signal.append(y)
        signal = pd.Series(signal).interpolate(limit_direction="both").values
        return h - signal

    fhr_signal = extract_signal(fhr_img)
    uc_signal = extract_signal(uc_img)

    px_per_cm = height / 10.0
    bpm_per_px = bpm_per_cm / px_per_cm
    toco_per_px = toco_per_cm / px_per_cm
    fhr_signal = fhr_min_line + fhr_signal * bpm_per_px
    uc_signal = uc_signal * toco_per_px
    px_per_sec = (paper_speed_cm_min / 60.0) * px_per_cm
    time_axis = np.arange(len(fhr_signal)) / px_per_sec

    return fhr_signal, uc_signal, time_axis

# (all your functions stay exactly the same — not repeated here to save space)
# extract_ctg_signals()
# compute_model_features()
def compute_model_features(fhr_signal, uc_signal, time_axis):
    """Compute SisPorto-style features for CTG classification."""
    features = {}
    duration = time_axis[-1] - time_axis[0]
    baseline = np.mean(fhr_signal)
    fhr_diff = np.diff(fhr_signal)

    # --- Accelerations ---
    accel_count = 0
    in_accel = False
    start_idx = None
    for i in range(len(fhr_signal)):
        if fhr_signal[i] > baseline + 5:
            if not in_accel:
                in_accel = True
                start_idx = i
        else:
            if in_accel:
                dur = time_axis[i - 1] - time_axis[start_idx]
                amp = np.max(fhr_signal[start_idx:i]) - baseline
                if dur >= 5 and amp >= 5:
                    accel_count += 1
                in_accel = False
    features["Accelerations (SisPorto)"] = round(accel_count / (duration / 600.0), 2)

    # --- Uterine contractions & fetal movements ---
    uc_peaks, _ = find_peaks(uc_signal, height=15, distance=int(30 * (len(time_axis) / duration)))
    fm_events, _ = find_peaks(np.diff(uc_signal), height=10, distance=int(5 * (len(time_axis) / duration)))
    features["Uterine contractions (SisPorto)"] = round(len(uc_peaks) / (duration / 600.0), 2)
    features["Fetal movements (SisPorto)"] = round(len(fm_events) / (duration / 600.0), 2)

    # --- Decelerations ---
    dips, _ = find_peaks(-fhr_signal, height=-(baseline - 15), distance=int(15))
    severe, _ = find_peaks(-fhr_signal, height=-(baseline - 25), distance=int(15))
    prolonged = [i for i in range(1, len(dips)) if (dips[i] - dips[i - 1]) > 120]
    repetitive = np.sum(np.diff(dips) < 60)
    features["Light decelerations (raw)"] = round(len(dips) / (duration / 600.0), 2)
    features["Severe decelerations (raw)"] = round(len(severe) / (duration / 600.0), 2)
    features["Prolonged decelerations (raw)"] = round(len(prolonged) / (duration / 600.0), 2)
    features["Repetitive decelerations (raw)"] = round(repetitive / (duration / 600.0), 2)

    # --- Baseline & variability ---
    features["Baseline value (SisPorto)"] = round(float(baseline), 2)
    features["Percentage time with abnormal short-term variability (SisPorto)"] = round(
        np.sum(np.abs(fhr_diff) > 25) / len(fhr_diff), 2
    )
    features["Mean value of short-term variability (SisPorto)"] = round(np.mean(np.abs(fhr_diff)), 2)
    features["Percentage time with abnormal long-term variability (SisPorto)"] = round(
        np.sum(np.abs(fhr_signal - baseline) > 20) / len(fhr_signal), 2
    )
    features["Mean value of long-term variability (SisPorto)"] = round(np.std(fhr_signal), 2)

    # --- Histogram ---
    hist, bins = np.histogram(fhr_signal, bins=10)
    features["Histogram width"] = round(bins[-1] - bins[0], 2)
    features["Histogram minimum frequency"] = round(np.min(fhr_signal), 2)
    features["Histogram maximum frequency"] = round(np.max(fhr_signal), 2)
    features["Number of histogram peaks"] = int(np.max(hist))
    features["Number of histogram zeros"] = int(np.sum(hist == 0))
    features["Histogram mode"] = round(bins[np.argmax(hist)], 2)
    features["Histogram mean"] = round(np.mean(fhr_signal), 2)
    features["Histogram median"] = round(np.median(fhr_signal), 2)
    features["Histogram variance"] = round(np.var(fhr_signal), 2)
    features["Histogram tendency (-1=left asymmetric; 0=symmetric; 1=right asymmetric)"] = round(
        fhr_signal[-1] - fhr_signal[0], 2
    )

    for col in model_ctg_class.feature_names_in_:
        if col not in features:
            features[col] = 0

    return features


# =======================================================
# ROUTES
# =======================================================
@app.get("/")
def home():
    return {"message": "CTG AI Prediction API is running 🚀"}

# -------------------------------------------------------
# PREDICT & SAVE (NOW WITH NYCKEL CTG DETECTION)
# -------------------------------------------------------
@app.post("/predict/")
async def predict_ctg(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ==================================================
        # ⭐ 1. RUN NYCKEL FIRST — CTG Detection
        # ==================================================
        files = {"data": open(tmp_path, "rb")}
        headers = {"Authorization": f"Bearer {NYCKEL_ACCESS_TOKEN}"}

        nyckel_response = requests.post(NYCKEL_URL, files=files, headers=headers)
        nyckel_data = nyckel_response.json()

        nyckel_label = nyckel_data.get("label", "").lower()

        # If NOT CTG → return immediately
        if "non" in nyckel_label or "other" in nyckel_label:
            return {
                "isCTG": False,
                "nyckelLabel": nyckel_data.get("label", "Unknown"),
                "confidence": nyckel_data.get("confidence", 0),
                "message": "Uploaded image is NOT a CTG scan."
            }

        # ==================================================
        # ⭐ If CTG detected → run your original pipeline
        # ==================================================
        fhr, uc, t = extract_ctg_signals(tmp_path)
        features = compute_model_features(fhr, uc, t)
        df = pd.DataFrame([features])[model_ctg_class.feature_names_in_]
        pred = model_ctg_class.predict(df)[0]
        label = {1: "Normal", 2: "Suspect", 3: "Pathologic"}.get(pred, "Unknown")

        # --- Upload to Cloudinary ---
        upload_result = cloudinary.uploader.upload(tmp_path, folder="drukhealth_ctg")
        image_url = upload_result.get("secure_url")

        # --- Store in MongoDB ---
        record = {
            "timestamp": datetime.utcnow() + timedelta(hours=6),
            "ctgDetected": label,
            "features": features,
            "imageUrl": image_url,
            "nyckelLabel": nyckel_data.get("label"),
            "nyckelConfidence": nyckel_data.get("confidence")
        }
        result = ctg_collection.insert_one(record)

        return {
            "isCTG": True,
            "nyckelLabel": nyckel_data.get("label"),
            "confidence": nyckel_data.get("confidence"),
            "prediction": int(pred),
            "label": label,
            "features": features,
            "imageUrl": image_url,
            "record_id": str(result.inserted_id),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# -------------------------------------------------------
# GET ALL RECORDS (unchanged)
# -------------------------------------------------------
@app.get("/records")
def get_records():
    try:
        records = list(ctg_collection.find().sort("timestamp", -1))
        formatted = []
        for rec in records:
            if rec.get("features") and rec.get("imageUrl"):
                formatted.append({
                    "id": str(rec["_id"]),
                    "timestamp": rec.get("timestamp"),
                    "ctgDetected": rec.get("ctgDetected", "Unknown"),
                    "features": rec.get("features", {}),
                    "imageUrl": rec.get("imageUrl", "")
                })
        return {"records": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------
# DELETE RECORD (unchanged)
# -------------------------------------------------------
@app.delete("/records/{record_id}")
def delete_record(record_id: str):
    try:
        result = ctg_collection.delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"detail": "Record deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------
# DASHBOARD ANALYSIS (unchanged)
# -------------------------------------------------------
@app.get("/api/analysis")
def get_analysis():
    records = list(ctg_collection.find({}, {"_id": 0}))
    if not records:
        return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    pivot = df.pivot_table(index="date", columns="ctgDetected", aggfunc="size", fill_value=0).reset_index()
    pivot = pivot.rename_axis(None, axis=1)
    pivot = pivot.rename(columns={"Normal": "N", "Suspect": "S", "Pathologic": "P"})
    time_series = pivot.to_dict(orient="records")

    counts = Counter(df["ctgDetected"])
    nspStats = {
        "Normal": int(counts.get("Normal", 0)),
        "Suspect": int(counts.get("Suspect", 0)),
        "Pathologic": int(counts.get("Pathologic", 0)),
    }

    return {"predictions": time_series, "nspStats": nspStats}


# =======================================================
# Run server
# =======================================================
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
