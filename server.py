# from fastapi import FastAPI, File, UploadFile, HTTPException, Response
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
# import cloudinary
# import cloudinary.uploader
# from dotenv import load_dotenv
# import tensorflow as tf
# import logging

# # ------------------------------------------------------------
# # ENV + LOGGING
# # ------------------------------------------------------------
# load_dotenv()
# logging.basicConfig(level=logging.INFO)

# # ------------------------------------------------------------
# # FASTAPI APP
# # ------------------------------------------------------------
# app = FastAPI(title="Druk Health CTG – 5 Feature Model")

# # CORS for both local + Vercel + Render
# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
#     "https://drukhealthfrontend.vercel.app",
#     "https://fastapi-backend-yrc0.onrender.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------------------------------------------------
# # HEALTH CHECK (Render HEAD + GET)
# # ------------------------------------------------------------
# @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
# def health_check():
#     return Response(
#         content='{"status": "ok"}',
#         media_type="application/json"
#     )

# # ------------------------------------------------------------
# # MongoDB
# # ------------------------------------------------------------
# MONGO_URI = os.getenv(
#     "MONGO_URI",
#     "mongodb+srv://12220045gcit:Kunzang1234@cluster0.rskaemg.mongodb.net/drukhealth?retryWrites=true&w=majority"
# )

# client = MongoClient(MONGO_URI)
# db = client["drukhealth"]
# ctg_collection = db["ctgscans"]

# # ------------------------------------------------------------
# # Cloudinary
# # ------------------------------------------------------------
# cloudinary.config(
#     cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b"),
#     api_key=os.getenv("CLOUDINARY_API_KEY", "522272821951884"),
#     api_secret=os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")
# )

# # ------------------------------------------------------------
# # Load CTG 5-Feature Model
# # ------------------------------------------------------------
# MODEL_5F_PATH = "ctg_5feature_model.pkl"

# if not os.path.exists(MODEL_5F_PATH):
#     raise FileNotFoundError("❌ Model missing: ctg_5feature_model.pkl")

# model_5f = joblib.load(MODEL_5F_PATH)
# logging.info("✅ CTG 5-Feature Decision Tree Model Loaded")

# # ------------------------------------------------------------
# # Load CNN Model
# # ------------------------------------------------------------
# CNN_PATH = "CTG_vs_NonCTG(1).keras"
# binary_ctg_model = tf.keras.models.load_model(CNN_PATH)
# logging.info("✅ CTG vs Non-CTG CNN Loaded")

# # ======================================================================
# # ===================== IMAGE → SIGNAL PROCESSING ======================
# # ======================================================================
# def classify_ctg_or_nonctg(img_path):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#     arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     arr = np.expand_dims(arr, 0)
#     score = float(binary_ctg_model.predict(arr)[0][0])
#     return ("CTG" if score > 0.5 else "Non-CTG"), score

# def detect_grid(gray):
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     th = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV, 21, 7
#     )
#     h, w = th.shape

#     horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
#     hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)

#     rows = np.where(np.sum(hl, axis=1) > np.percentile(np.sum(hl, axis=1), 95))[0]
#     if len(rows) < 2:
#         return None

#     spacing = np.diff(rows)
#     spacing = spacing[spacing > 3]
#     if len(spacing) == 0:
#         return None

#     return int(np.median(spacing) * 5)

# def extract_trace(roi):
#     blur = cv2.GaussianBlur(roi, (3, 3), 0)
#     th = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV, 15, 5
#     )
#     h, w = th.shape
#     ys = []

#     for x in range(w):
#         idx = np.where(th[:, x] > 0)[0]
#         ys.append(np.median(idx) if len(idx) else np.nan)

#     ys = np.array(ys)
#     xs = np.arange(len(ys))

#     ys[np.isnan(ys)] = np.interp(xs[np.isnan(ys)], xs[~np.isnan(ys)], ys[~np.isnan(ys)])

#     return ys

# def extract_ctg_signals(path):
#     img = cv2.imread(path)
#     h, w = img.shape[:2]

#     if h > w:
#         img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#         h, w = img.shape[:2]

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     px_large = detect_grid(gray) or 180

#     top = int(h * 0.05)
#     mid = int(h * 0.45)
#     bottom = int(h * 0.95)

#     fhr_y = extract_trace(gray[top:mid, :]) + top
#     uc_y = extract_trace(gray[mid:bottom, :])

#     bpm_pp = 30 / px_large
#     ref_y = (top + mid) / 2

#     fhr_bpm = 140 - (fhr_y - ref_y) * bpm_pp
#     fhr_bpm = medfilt(fhr_bpm, 7)
#     fhr_bpm = np.clip(fhr_bpm, 40, 220)

#     sec_per_px = 60 / px_large
#     fs = 1 / sec_per_px

#     uc_norm = (uc_y - np.min(uc_y)) / (np.ptp(uc_y) + 1e-6)

#     return fhr_bpm, uc_norm, fs

# # ======================================================================
# # ======================= 5-FEATURE EXTRACTION ==========================
# # ======================================================================
# def compute_features_5f(fhr, uc, fs):
#     samples_per_min = max(1, int(fs * 60))
#     baseline = float(np.median(fhr[:samples_per_min]))
#     std_val = float(np.std(fhr[:samples_per_min]))

#     if std_val < 5:
#         variability = 0
#     elif std_val < 10:
#         variability = 1
#     elif std_val < 25:
#         variability = 2
#     else:
#         variability = 3

#     samples_20 = samples_per_min * 20
#     fwin = fhr[:samples_20]

#     def detect(kind):
#         amp = 15
#         dur = int(fs * 15)
#         N = len(fwin)
#         c = 0
#         i = 0

#         while i < N:
#             if kind == "accel" and fwin[i] > baseline + amp:
#                 s = i
#                 while i < N and fwin[i] > baseline + 5:
#                     i += 1
#                 if i - s >= dur:
#                     c += 1

#             elif kind == "decel" and fwin[i] < baseline - amp:
#                 s = i
#                 while i < N and fwin[i] < baseline - 5:
#                     i += 1
#                 if i - s >= dur:
#                     c += 1

#             else:
#                 i += 1
#         return c

#     acceleration = detect("accel")
#     deceleration = detect("decel")

#     peaks, _ = find_peaks(uc, height=np.mean(uc) + 0.2)
#     uterine = len(peaks)

#     return {
#         "acceleration": acceleration,
#         "deceleration": deceleration,
#         "baseline": float(round(baseline, 2)),
#         "uterine_contraction": uterine,
#         "variability": variability,
#     }

# # ======================================================================
# # ============================= PREDICT ================================
# # ======================================================================
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         tmp.write(contents)
#         path = tmp.name

#     try:
#         label_bin, score = classify_ctg_or_nonctg(path)

#         if label_bin == "Non-CTG":
#             up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
#             return {
#                 "isCTG": False,
#                 "message": "Not a CTG graph",
#                 "confidence": score,
#                 "imageUrl": up["secure_url"],
#             }

#         fhr, uc, fs = extract_ctg_signals(path)
#         features = compute_features_5f(fhr, uc, fs)

#         df = pd.DataFrame([features])
#         pred_num = int(model_5f.predict(df)[0])
#         label = {1: "Normal", 2: "Suspicious", 3: "Pathological"}[pred_num]

#         up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")

#         rec = {
#             "timestamp": datetime.utcnow() + timedelta(hours=6),
#             "isCTG": True,
#             "binaryConfidence": score,
#             "classNumeric": pred_num,
#             "labelClient": label,
#             "features": features,
#             "imageUrl": up["secure_url"],
#         }

#         res = ctg_collection.insert_one(rec)

#         return {
#             "isCTG": True,
#             "label": label,
#             "numeric": pred_num,
#             "features": features,
#             "confidence": score,
#             "imageUrl": up["secure_url"],
#             "record_id": str(res.inserted_id),
#         }

#     finally:
#         try:
#             os.remove(path)
#         except:
#             pass

# # ======================================================================
# # ========================== RECORD ENDPOINTS ===========================
# # ======================================================================
# @app.get("/records")
# def get_records():
#     recs = list(ctg_collection.find().sort("timestamp", -1))
#     return {
#         "records": [
#             {
#                 "id": str(r["_id"]),
#                 "timestamp": r.get("timestamp"),
#                 "isCTG": r.get("isCTG"),
#                 "binaryConfidence": r.get("binaryConfidence"),
#                 "labelClient": r.get("labelClient"),
#                 "features": r.get("features"),
#                 "imageUrl": r.get("imageUrl"),
#             }
#             for r in recs
#         ]
#     }

# @app.delete("/records/{record_id}")
# def delete_record(record_id: str):
#     r = ctg_collection.delete_one({"_id": ObjectId(record_id)})
#     if r.deleted_count == 0:
#         raise HTTPException(404, "Record not found")
#     return {"detail": "Record deleted"}

# # ======================================================================
# # ========================== ANALYSIS ==================================
# # ======================================================================
# @app.get("/api/analysis")
# def analysis():
#     data = list(ctg_collection.find({}, {"_id": 0}))

#     if not data:
#         return {
#             "predictions": [],
#             "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0},
#         }

#     df = pd.DataFrame(data)
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

#     pivot = df.pivot_table(
#         index="date",
#         columns="classNumeric",
#         aggfunc="size",
#         fill_value=0
#     )

#     counts = df["classNumeric"].value_counts()

#     return {
#         "predictions": pivot.reset_index().to_dict(orient="records"),
#         "nspStats": {
#             "Normal": int(counts.get(1, 0)),
#             "Suspect": int(counts.get(2, 0)),
#             "Pathologic": int(counts.get(3, 0)),
#         },
#     }

# # ======================================================================
# # ====================== FEATURE IMPORTANCE =============================
# # ======================================================================
# @app.get("/api/feature-importance")
# def feature_importance():
#     names = [
#         "acceleration",
#         "deceleration",
#         "baseline",
#         "uterine_contraction",
#         "variability",
#     ]

#     importances = model_5f.feature_importances_

#     return {
#         "feature_importance": {
#             n: float(v) for n, v in zip(names, importances)
#         }
#     }

# # ======================================================================
# # =============================== RUN ==================================
# # ======================================================================
# if __name__ == "__main__":
#     import uvicorn

#     port = int(os.environ.get("PORT", 9000))  # 9000 local, $PORT in Render

#     uvicorn.run(
#         "server:app",
#         host="0.0.0.0",
#         port=port,
#         reload=False  # Required for Windows / Render stability
#     )


































# from fastapi import FastAPI, File, UploadFile, HTTPException, Response
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
# import cloudinary
# import cloudinary.uploader
# from dotenv import load_dotenv
# import tensorflow as tf
# import logging

# # ------------------------------------------------------------
# # ENV + LOGGING
# # ------------------------------------------------------------
# load_dotenv()
# logging.basicConfig(level=logging.INFO)

# # ------------------------------------------------------------
# # FASTAPI APP
# # ------------------------------------------------------------
# app = FastAPI(title="Druk Health CTG – 5 Feature Model")

# # CORS for both local + Vercel + Render
# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
#     "https://drukhealthfrontend.vercel.app",
#     "https://fastapi-backend-yrc0.onrender.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------------------------------------------------
# # HEALTH CHECK (Render HEAD + GET)
# # ------------------------------------------------------------
# @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
# def health_check():
#     return Response(
#         content='{"status": "ok"}',
#         media_type="application/json"
#     )

# # ------------------------------------------------------------
# # MongoDB
# # ------------------------------------------------------------
# MONGO_URI = os.getenv(
#     "MONGO_URI",
#     "mongodb+srv://12220045gcit:Kunzang1234@cluster0.rskaemg.mongodb.net/drukhealth?retryWrites=true&w=majority"
# )

# client = MongoClient(MONGO_URI)
# db = client["drukhealth"]
# ctg_collection = db["ctgscans"]

# # ------------------------------------------------------------
# # Cloudinary
# # ------------------------------------------------------------
# cloudinary.config(
#     cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b"),
#     api_key=os.getenv("CLOUDINARY_API_KEY", "522272821951884"),
#     api_secret=os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")
# )

# # ------------------------------------------------------------
# # Load CTG 5-Feature Model
# # ------------------------------------------------------------
# MODEL_5F_PATH = "ctg_5feature_model.pkl"

# if not os.path.exists(MODEL_5F_PATH):
#     raise FileNotFoundError("❌ Model missing: ctg_5feature_model.pkl")

# model_5f = joblib.load(MODEL_5F_PATH)
# logging.info("✅ CTG 5-Feature Decision Tree Model Loaded")

# # ------------------------------------------------------------
# # Load CNN Model
# # ------------------------------------------------------------
# CNN_PATH = "CTG_vs_NonCTG(1).keras"
# binary_ctg_model = tf.keras.models.load_model(CNN_PATH)
# logging.info("✅ CTG vs Non-CTG CNN Loaded")

# # ======================================================================
# # ===================== IMAGE → SIGNAL PROCESSING ======================
# # ======================================================================
# def classify_ctg_or_nonctg(img_path):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#     arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     arr = np.expand_dims(arr, 0)
#     score = float(binary_ctg_model.predict(arr)[0][0])
#     return ("CTG" if score > 0.5 else "Non-CTG"), score

# def detect_grid(gray):
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     th = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV, 21, 7
#     )
#     h, w = th.shape

#     horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
#     hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)

#     rows = np.where(np.sum(hl, axis=1) > np.percentile(np.sum(hl, axis=1), 95))[0]
#     if len(rows) < 2:
#         return None

#     spacing = np.diff(rows)
#     spacing = spacing[spacing > 3]
#     if len(spacing) == 0:
#         return None

#     return int(np.median(spacing) * 5)

# def extract_trace(roi):
#     blur = cv2.GaussianBlur(roi, (3, 3), 0)
#     th = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV, 15, 5
#     )
#     h, w = th.shape
#     ys = []

#     for x in range(w):
#         idx = np.where(th[:, x] > 0)[0]
#         ys.append(np.median(idx) if len(idx) else np.nan)

#     ys = np.array(ys)
#     xs = np.arange(len(ys))

#     ys[np.isnan(ys)] = np.interp(xs[np.isnan(ys)], xs[~np.isnan(ys)], ys[~np.isnan(ys)])

#     return ys

# def extract_ctg_signals(path):
#     img = cv2.imread(path)
#     h, w = img.shape[:2]

#     if h > w:
#         img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#         h, w = img.shape[:2]

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     px_large = detect_grid(gray) or 180

#     top = int(h * 0.05)
#     mid = int(h * 0.45)
#     bottom = int(h * 0.95)

#     fhr_y = extract_trace(gray[top:mid, :]) + top
#     uc_y = extract_trace(gray[mid:bottom, :])

#     bpm_pp = 30 / px_large
#     ref_y = (top + mid) / 2

#     fhr_bpm = 140 - (fhr_y - ref_y) * bpm_pp
#     fhr_bpm = medfilt(fhr_bpm, 7)
#     fhr_bpm = np.clip(fhr_bpm, 40, 220)

#     sec_per_px = 60 / px_large
#     fs = 1 / sec_per_px

#     uc_norm = (uc_y - np.min(uc_y)) / (np.ptp(uc_y) + 1e-6)

#     return fhr_bpm, uc_norm, fs

# # ======================================================================
# # ======================= 5-FEATURE EXTRACTION ==========================
# # ======================================================================
# def compute_features_5f(fhr, uc, fs):
#     samples_per_min = max(1, int(fs * 60))
#     baseline = float(np.median(fhr[:samples_per_min]))
#     std_val = float(np.std(fhr[:samples_per_min]))

#     if std_val < 5:
#         variability = 0
#     elif std_val < 10:
#         variability = 1
#     elif std_val < 25:
#         variability = 2
#     else:
#         variability = 3

#     samples_20 = samples_per_min * 20
#     fwin = fhr[:samples_20]

#     def detect(kind):
#         amp = 15
#         dur = int(fs * 15)
#         N = len(fwin)
#         c = 0
#         i = 0

#         while i < N:
#             if kind == "accel" and fwin[i] > baseline + amp:
#                 s = i
#                 while i < N and fwin[i] > baseline + 5:
#                     i += 1
#                 if i - s >= dur:
#                     c += 1

#             elif kind == "decel" and fwin[i] < baseline - amp:
#                 s = i
#                 while i < N and fwin[i] < baseline - 5:
#                     i += 1
#                 if i - s >= dur:
#                     c += 1

#             else:
#                 i += 1
#         return c

#     acceleration = detect("accel")
#     deceleration = detect("decel")

#     peaks, _ = find_peaks(uc, height=np.mean(uc) + 0.2)
#     uterine = len(peaks)

#     return {
#         "acceleration": acceleration,
#         "deceleration": deceleration,
#         "baseline": float(round(baseline, 2)),
#         "uterine_contraction": uterine,
#         "variability": variability,
#     }

# # ======================================================================
# # ======================= INTERPRETATION LAYER ===========================
# # ======================================================================
# def interpret_ctg(features):
#     """
#     Map numeric features to concise, medically phrased interpretations.
#     Returns a dict of interpretation sentences keyed to each feature.
#     """
#     baseline = features.get("baseline", None)
#     variability = features.get("variability", None)
#     acc = features.get("acceleration", 0)
#     dec = features.get("deceleration", 0)
#     uc = features.get("uterine_contraction", 0)

#     interp = {}

#     # Baseline interpretation
#     if baseline is None:
#         interp["baseline_interpretation"] = "Baseline unavailable."
#     else:
#         if 110 <= baseline <= 160:
#             interp["baseline_interpretation"] = (
#                 f"Baseline {baseline} bpm: within normal range (110–160 bpm), indicating appropriate fetal autonomic regulation."
#             )
#         elif baseline < 110:
#             interp["baseline_interpretation"] = (
#                 f"Baseline {baseline} bpm: bradycardia (<110 bpm) — consider maternal factors, fetal hypoxia or further urgent assessment."
#             )
#         else:
#             interp["baseline_interpretation"] = (
#                 f"Baseline {baseline} bpm: tachycardia (>160 bpm) — consider maternal fever, infection, fetal distress, or medications."
#             )

#     # Variability interpretation (0: reduced, 1: borderline, 2: normal, 3: saltatory)
#     if variability is None:
#         interp["variability_interpretation"] = "Variability unavailable."
#     else:
#         if variability == 0:
#             interp["variability_interpretation"] = (
#                 "Reduced variability (<5 bpm): may indicate fetal hypoxia, sleep cycle, or depressant medications — correlate clinically."
#             )
#         elif variability == 1:
#             interp["variability_interpretation"] = (
#                 "Borderline variability (5–10 bpm): warrants closer observation and correlation with accelerations and clinical context."
#             )
#         elif variability == 2:
#             interp["variability_interpretation"] = (
#                 "Normal variability (10–25 bpm): reassuring for fetal oxygenation and neurological responsiveness."
#             )
#         else:
#             interp["variability_interpretation"] = (
#                 "Marked/saltatory variability (>25 bpm): uncommon and may represent artifact or fetal compromise depending on context."
#             )

#     # Accelerations
#     if acc > 0:
#         interp["acceleration_interpretation"] = (
#             f"{acc} acceleration(s) detected: presence of accelerations is reassuring and suggests intact fetal autonomic response."
#         )
#     else:
#         interp["acceleration_interpretation"] = (
#             "No accelerations detected in the analyzed window: absence of accelerations reduces reassurance — correlate with variability and clinical situation."
#         )

#     # Decelerations
#     if dec > 0:
#         interp["deceleration_interpretation"] = (
#             f"{dec} deceleration(s) detected: decelerations require morphological assessment (early/variable/late). "
#             "Late or repetitive decelerations may indicate uteroplacental insufficiency and need escalation."
#         )
#     else:
#         interp["deceleration_interpretation"] = "No decelerations detected in the analyzed window."

#     # Uterine contractions
#     interp["contraction_interpretation"] = (
#         f"{uc} uterine contraction(s) detected in the analyzed window. Assess frequency and relation to any decelerations for clinical correlation."
#     )

#     # Overall brief advisory (conservative)
#     # Note: we intentionally do NOT override the model classification — this is an explanatory layer.
#     interp["clinical_advice"] = (
#         "Use these automated interpretations as supportive information only. Clinical decisions should integrate full-length CTG review, maternal/fetal history, and bedside assessment."
#     )

#     return interp

# # ======================================================================
# # ============================= PREDICT ================================
# # ======================================================================
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         tmp.write(contents)
#         path = tmp.name

#     try:
#         label_bin, score = classify_ctg_or_nonctg(path)

#         if label_bin == "Non-CTG":
#             up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
#             return {
#                 "isCTG": False,
#                 "message": "Not a CTG graph",
#                 "confidence": score,
#                 "imageUrl": up["secure_url"],
#             }

#         fhr, uc, fs = extract_ctg_signals(path)
#         features = compute_features_5f(fhr, uc, fs)

#         # --- new: interpretation based on features ---
#         interpretation = interpret_ctg(features)

#         df = pd.DataFrame([features])
#         pred_num = int(model_5f.predict(df)[0])
#         label = {1: "Normal", 2: "Suspicious", 3: "Pathological"}[pred_num]

#         up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")

#         rec = {
#             "timestamp": datetime.utcnow() + timedelta(hours=6),
#             "isCTG": True,
#             "binaryConfidence": score,
#             "classNumeric": pred_num,
#             "labelClient": label,
#             "features": features,
#             "interpretation": interpretation,   # saved to DB
#             "imageUrl": up["secure_url"],
#         }

#         res = ctg_collection.insert_one(rec)

#         return {
#             "isCTG": True,
#             "label": label,
#             "numeric": pred_num,
#             "features": features,
#             "interpretation": interpretation,   # returned in API response
#             "confidence": score,
#             "imageUrl": up["secure_url"],
#             "record_id": str(res.inserted_id),
#         }

#     finally:
#         try:
#             os.remove(path)
#         except:
#             pass

# # ======================================================================
# # ========================== RECORD ENDPOINTS ===========================
# # ======================================================================
# @app.get("/records")
# def get_records():
#     recs = list(ctg_collection.find().sort("timestamp", -1))
#     return {
#         "records": [
#             {
#                 "id": str(r["_id"]),
#                 "timestamp": r.get("timestamp"),
#                 "isCTG": r.get("isCTG"),
#                 "binaryConfidence": r.get("binaryConfidence"),
#                 "labelClient": r.get("labelClient"),
#                 "features": r.get("features"),
#                 "interpretation": r.get("interpretation"),
#                 "imageUrl": r.get("imageUrl"),
#             }
#             for r in recs
#         ]
#     }

# @app.delete("/records/{record_id}")
# def delete_record(record_id: str):
#     r = ctg_collection.delete_one({"_id": ObjectId(record_id)})
#     if r.deleted_count == 0:
#         raise HTTPException(404, "Record not found")
#     return {"detail": "Record deleted"}

# # ======================================================================
# # ========================== ANALYSIS =================================
# # ======================================================================
# @app.get("/api/analysis")
# def analysis():
#     data = list(ctg_collection.find({}, {"_id": 0}))

#     if not data:
#         return {
#             "predictions": [],
#             "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0},
#         }

#     df = pd.DataFrame(data)
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

#     pivot = df.pivot_table(
#         index="date",
#         columns="classNumeric",
#         aggfunc="size",
#         fill_value=0
#     )

#     counts = df["classNumeric"].value_counts()

#     return {
#         "predictions": pivot.reset_index().to_dict(orient="records"),
#         "nspStats": {
#             "Normal": int(counts.get(1, 0)),
#             "Suspect": int(counts.get(2, 0)),
#             "Pathologic": int(counts.get(3, 0)),
#         },
#     }

# # ======================================================================
# # ====================== FEATURE IMPORTANCE =============================
# # ======================================================================
# @app.get("/api/feature-importance")
# def feature_importance():
#     names = [
#         "acceleration",
#         "deceleration",
#         "baseline",
#         "uterine_contraction",
#         "variability",
#     ]

#     importances = model_5f.feature_importances_

#     return {
#         "feature_importance": {
#             n: float(v) for n, v in zip(names, importances)
#         }
#     }

# # ======================================================================
# # =============================== RUN ==================================
# # ======================================================================
# if __name__ == "__main__":
#     import uvicorn

#     port = int(os.environ.get("PORT", 9000))  # 9000 local, $PORT in Render

#     uvicorn.run(
#         "server:app",
#         host="0.0.0.0",
#         port=port,
#         reload=False  # Required for Windows / Render stability
#     )









from fastapi import FastAPI, File, UploadFile, HTTPException, Response
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
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import tensorflow as tf
import logging

# ------------------------------------------------------------
# ENV + LOGGING
# ------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(title="Druk Health CTG – 5 Feature Model")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://drukhealthfrontend.vercel.app",
    "https://fastapi-backend-yrc0.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def health_check():
    return Response(content='{"status": "ok"}', media_type="application/json")

# ------------------------------------------------------------
# MongoDB
# ------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["drukhealth"]
ctg_collection = db["ctgscans"]

# ------------------------------------------------------------
# Cloudinary
# ------------------------------------------------------------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dgclndz9b"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "522272821951884"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "gGICVeYwIKD02hW0weemvE1Ju98")
)

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
MODEL_5F_PATH = "ctg_5feature_model.pkl"
if not os.path.exists(MODEL_5F_PATH):
    raise FileNotFoundError("❌ Model missing: ctg_5feature_model.pkl")
model_5f = joblib.load(MODEL_5F_PATH)
logging.info("✅ CTG 5-Feature Decision Tree Model Loaded")

CNN_PATH = "CTG_vs_NonCTG(1).keras"
binary_ctg_model = tf.keras.models.load_model(CNN_PATH)
logging.info("✅ CTG vs Non-CTG CNN Loaded")

# ======================================================================
# IMAGE → SIGNAL PROCESSING
# ======================================================================
def classify_ctg_or_nonctg(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    score = float(binary_ctg_model.predict(arr)[0][0])
    return ("CTG" if score > 0.5 else "Non-CTG"), score

def detect_grid(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 7)
    h, w = th.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)
    rows = np.where(np.sum(hl, axis=1) > np.percentile(np.sum(hl, axis=1), 95))[0]
    if len(rows) < 2:
        return None
    spacing = np.diff(rows)
    spacing = spacing[spacing > 3]
    if len(spacing) == 0:
        return None
    return int(np.median(spacing) * 5)

def extract_trace(roi):
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    h, w = th.shape
    ys = []
    for x in range(w):
        idx = np.where(th[:, x] > 0)[0]
        ys.append(np.median(idx) if len(idx) else np.nan)
    ys = np.array(ys)
    xs = np.arange(len(ys))
    ys[np.isnan(ys)] = np.interp(xs[np.isnan(ys)], xs[~np.isnan(ys)], ys[~np.isnan(ys)])
    return ys

def extract_ctg_signals(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    px_large = detect_grid(gray) or 180
    top = int(h * 0.05)
    mid = int(h * 0.45)
    bottom = int(h * 0.95)
    fhr_y = extract_trace(gray[top:mid, :]) + top
    uc_y = extract_trace(gray[mid:bottom, :])
    bpm_pp = 30 / px_large
    ref_y = (top + mid) / 2
    fhr_bpm = 140 - (fhr_y - ref_y) * bpm_pp
    fhr_bpm = medfilt(fhr_bpm, 7)
    fhr_bpm = np.clip(fhr_bpm, 40, 220)
    sec_per_px = 60 / px_large
    fs = 1 / sec_per_px
    uc_norm = (uc_y - np.min(uc_y)) / (np.ptp(uc_y) + 1e-6)
    return fhr_bpm, uc_norm, fs

# ======================================================================
# 5-FEATURE EXTRACTION
# ======================================================================
def compute_features_5f(fhr, uc, fs):
    samples_per_min = max(1, int(fs * 60))
    baseline = float(np.median(fhr[:samples_per_min]))
    std_val = float(np.std(fhr[:samples_per_min]))
    if std_val < 5:
        variability = 0
    elif std_val < 10:
        variability = 1
    elif std_val < 25:
        variability = 2
    else:
        variability = 3
    samples_20 = samples_per_min * 20
    fwin = fhr[:samples_20]

    def detect(kind):
        amp = 15
        dur = int(fs * 15)
        N = len(fwin)
        c = 0
        i = 0
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
    peaks, _ = find_peaks(uc, height=np.mean(uc) + 0.2)
    uterine = len(peaks)

    return {
        "acceleration": acceleration,
        "deceleration": deceleration,
        "baseline": float(round(baseline, 2)),
        "uterine_contraction": uterine,
        "variability": variability,
    }

# ======================================================================
# INTERPRETATION LAYER
# ======================================================================
def interpret_ctg(features):
    baseline = features.get("baseline", None)
    variability = features.get("variability", None)
    acc = features.get("acceleration", 0)
    dec = features.get("deceleration", 0)
    uc = features.get("uterine_contraction", 0)

    interp = {}

    if baseline is None:
        interp["baseline_interpretation"] = "Baseline unavailable."
    else:
        if 110 <= baseline <= 160:
            interp["baseline_interpretation"] = f"Baseline {baseline} bpm: within normal range (110–160 bpm), indicating appropriate fetal autonomic regulation."
        elif baseline < 110:
            interp["baseline_interpretation"] = f"Baseline {baseline} bpm: bradycardia (<110 bpm) — consider maternal factors, fetal hypoxia or further urgent assessment."
        else:
            interp["baseline_interpretation"] = f"Baseline {baseline} bpm: tachycardia (>160 bpm) — consider maternal fever, infection, fetal distress, or medications."

    if variability is None:
        interp["variability_interpretation"] = "Variability unavailable."
    else:
        if variability == 0:
            interp["variability_interpretation"] = "Reduced variability (<5 bpm): may indicate fetal hypoxia, sleep cycle, or depressant medications — correlate clinically."
        elif variability == 1:
            interp["variability_interpretation"] = "Borderline variability (5–10 bpm): warrants closer observation and correlation with accelerations and clinical context."
        elif variability == 2:
            interp["variability_interpretation"] = "Normal variability (10–25 bpm): reassuring for fetal oxygenation and neurological responsiveness."
        else:
            interp["variability_interpretation"] = "Marked/saltatory variability (>25 bpm): uncommon and may represent artifact or fetal compromise depending on context."

    interp["acceleration_interpretation"] = (
        f"{acc} acceleration(s) detected: presence of accelerations is reassuring and suggests intact fetal autonomic response." 
        if acc > 0 else
        "No accelerations detected in the analyzed window: absence of accelerations reduces reassurance — correlate with variability and clinical situation."
    )

    interp["deceleration_interpretation"] = (
        f"{dec} deceleration(s) detected: decelerations require morphological assessment (early/variable/late). Late or repetitive decelerations may indicate uteroplacental insufficiency and need escalation."
        if dec > 0 else
        "No decelerations detected in the analyzed window."
    )

    interp["contraction_interpretation"] = f"{uc} uterine contraction(s) detected in the analyzed window. Assess frequency and relation to any decelerations for clinical correlation."

    interp["clinical_advice"] = "Use these automated interpretations as supportive information only. Clinical decisions should integrate full-length CTG review, maternal/fetal history, and bedside assessment."

    return interp

# ======================================================================
# PREDICT ENDPOINT
# ======================================================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        path = tmp.name

    try:
        label_bin, score = classify_ctg_or_nonctg(path)

        if label_bin == "Non-CTG":
            up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")
            return {
                "isCTG": False,
                "message": "Not a CTG graph",
                "confidence": score,
                "imageUrl": up["secure_url"],
            }

        fhr, uc, fs = extract_ctg_signals(path)
        features = compute_features_5f(fhr, uc, fs)

        # --- interpretation logging ---
        interpretation = interpret_ctg(features)
        print("Interpretation generated:", interpretation)

        df = pd.DataFrame([features])
        pred_num = int(model_5f.predict(df)[0])
        label = {1: "Normal", 2: "Suspicious", 3: "Pathological"}[pred_num]

        up = cloudinary.uploader.upload(path, folder="drukhealth_ctg")

        rec = {
            "timestamp": datetime.utcnow() + timedelta(hours=6),
            "isCTG": True,
            "binaryConfidence": score,
            "classNumeric": pred_num,
            "labelClient": label,
            "features": features,
            "interpretation": interpretation,
            "imageUrl": up["secure_url"],
        }

        res = ctg_collection.insert_one(rec)
        print("Inserted record:", rec)

        return {
            "isCTG": True,
            "label": label,
            "numeric": pred_num,
            "features": features,
            "interpretation": interpretation,
            "confidence": score,
            "imageUrl": up["secure_url"],
            "record_id": str(res.inserted_id),
        }

    finally:
        try:
            os.remove(path)
        except:
            pass

# ======================================================================
# RECORD ENDPOINTS
# ======================================================================
@app.get("/records")
def get_records():
    recs = list(ctg_collection.find().sort("timestamp", -1))
    return {
        "records": [
            {
                "id": str(r["_id"]),
                "timestamp": r.get("timestamp"),
                "isCTG": r.get("isCTG"),
                "binaryConfidence": r.get("binaryConfidence"),
                "labelClient": r.get("labelClient"),
                "features": r.get("features"),
                "interpretation": r.get("interpretation"),
                "imageUrl": r.get("imageUrl"),
            }
            for r in recs
        ]
    }

@app.delete("/records/{record_id}")
def delete_record(record_id: str):
    r = ctg_collection.delete_one({"_id": ObjectId(record_id)})
    if r.deleted_count == 0:
        raise HTTPException(404, "Record not found")
    return {"detail": "Record deleted"}

# ======================================================================
# ANALYSIS ENDPOINTS
# ======================================================================
@app.get("/api/analysis")
def analysis():
    data = list(ctg_collection.find({}, {"_id": 0}))
    if not data:
        return {"predictions": [], "nspStats": {"Normal": 0, "Suspect": 0, "Pathologic": 0}}

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    pivot = df.pivot_table(index="date", columns="classNumeric", aggfunc="size", fill_value=0)
    counts = df["classNumeric"].value_counts()
    return {
        "predictions": pivot.reset_index().to_dict(orient="records"),
        "nspStats": {
            "Normal": int(counts.get(1, 0)),
            "Suspect": int(counts.get(2, 0)),
            "Pathologic": int(counts.get(3, 0)),
        },
    }

# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================
@app.get("/api/feature-importance")
def feature_importance():
    names = ["acceleration", "deceleration", "baseline", "uterine_contraction", "variability"]
    importances = model_5f.feature_importances_
    return {"feature_importance": {n: float(v) for n, v in zip(names, importances)}}

# ======================================================================
# RUN
# ======================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)