## Zhiwa CTG - AI Model Inference Server

### Purpose
The FastAPI Backend serves as the AI inference engine for the Zhiwa CTG system. It handles CTG image processing and classification using trained machine learning models to predict fetal health status as Normal, Suspicious, or Pathological.

### Features
- AI Model Inference: Loads and runs pre-trained ML models for CTG classification

- Image Processing: Handles CTG graph image preprocessing and feature extraction

- RESTful API: Provides clean endpoints for model predictions

- Fast Performance: Optimized for quick inference using FastAPI

### Architecture
#### Tech Stack
- Framework: FastAPI

- Machine Learning: Scikit-learn, OpenCV, NumPy

- Image Processing: OpenCV, PIL

### Project Structure

FASTAPI_BACKEND

│

├── clinical_aware_ctg_model.pkl

├── decision_tree_all_cardii_features.pkl

├── requirements.txt

└── server.py

#### File Descriptions:

- clinical_aware_ctg_model.pkl: Pre-trained clinical-aware CTG classification model

- decision_tree_all_cardii_features.pkl: Decision tree model with comprehensive cardiology features

- requirements.txt: Python dependencies and versions

- server.py: FastAPI application with model loading and prediction endpoints

### Installation Steps
#### Prerequisites
    - Python 3.8 or higher

    - pip (Python package manager)

##### Setup Instructions
1. Install dependencies
   
    - pip install -r requirements.txt

2. Run the server

    - python server.py

### API Documentation
#### Base URL

http://localhost:8000

#### Endpoints
Health Check
- GET /

- Response: Server status and available models

#### Model Prediction
- POST /predict

- Body: Form data with CTG image file

- Response: Prediction results with confidence scores

#### Model Information
- GET /models

- Response: List of loaded models and their specifications

#### Response Format

{
  "success": true,
  "prediction": "Normal",
  "confidence": 0.85,
  "model_used": "clinical_aware_ctg_model",
  "processing_time": 1.23
}

#### Requirements
The requirements.txt should contain:

  fastapi>=0.104.0
  uvicorn>=0.24.0
  python-multipart>=0.0.6
  opencv-python>=4.8.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  Pillow>=10.0.0
  joblib>=1.3.0
  pydantic>=2.0.0

### Model Information
#### Available Models
1. clinical_aware_ctg_model.pkl

  - Clinical-aware CTG classification model
  
  - Output: Normal, Suspicious, Pathological

2. decision_tree_all_cardii_features.pkl

  - Decision Tree classifier with cardiology features
  
  - Output: Fetal health classification

### API Integration
#### Integration with Main Backend
The Node.js backend sends CTG images to this server:

  const response = await fetch('http://localhost:8000/predict', 
  
  {
    method: 'POST',
    body: formData // containing CTG image
  });


  
  
