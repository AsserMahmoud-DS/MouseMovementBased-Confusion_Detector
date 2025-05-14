import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from typing import List

app = FastAPI(
    title="Confusion Detector API",
    description="API to predict user confusion from cursor movement data"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature engineering functions
def calculate_angle(x1, y1, x2, y2, x3, y3):
    v1 = [x2 - x1, y2 - y1]
    v2 = [x3 - x2, y3 - y2]
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def label_angle(angle):
    if np.isnan(angle):    
        return 0
    if angle < 30:
        return 0
    elif angle < 60:
        return 1
    elif angle < 90:
        return 2
    elif angle < 120:
        return 3
    elif angle < 150:
        return 4
    else:
        return 5

def extract_features(extracted, time_diff=0.05):
    extracted['distance_covered'] = 0.0
    extracted['idle_time'] = 0.0

    for i in range(1, len(extracted)):
        dx = extracted.loc[i, 'x'] - extracted.loc[i-1, 'x']
        dy = extracted.loc[i, 'y'] - extracted.loc[i-1, 'y']
        distance = np.sqrt(dx**2 + dy**2)
        extracted.loc[i, 'distance_covered'] = distance

        if extracted.loc[i, 'x'] == extracted.loc[i-1, 'x'] and extracted.loc[i, 'y'] == extracted.loc[i-1, 'y']:
            extracted.loc[i, 'idle_time'] = extracted.loc[i-1, 'idle_time'] + time_diff
        else:
            extracted.loc[i, 'idle_time'] = 0.0

    extracted['cursor_speed'] = extracted['distance_covered'] / time_diff
    extracted['acceleration'] = extracted['cursor_speed'] / time_diff

    angles = []
    for i in range(1, len(extracted) - 1):
        angle = calculate_angle(
            extracted.loc[i-1, 'x'], extracted.loc[i-1, 'y'],
            extracted.loc[i, 'x'], extracted.loc[i, 'y'],
            extracted.loc[i+1, 'x'], extracted.loc[i+1, 'y']
        )
        angles.append(angle)

    extracted = extracted.iloc[1:-1].copy()
    extracted['movement_angle'] = angles
    extracted['prev_movement_angle'] = [0] + angles[1:]

    extracted['angle_label'] = extracted['movement_angle'].apply(label_angle)
    extracted['prev_angle_label'] = extracted['prev_movement_angle'].apply(label_angle)
    return extracted

# Load model
model = XGBClassifier()
model.load_model("xgb_confusion_detector.model")

# Pydantic model for response
class PredictionResponse(BaseModel):
    Prediction: List[int]
    Confidence: List[List[float]]
    ConfusionRatio: float
    UserConfused: bool

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        raw_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        processed_df = extract_features(raw_df, time_diff=0.05)

        try:
            features_to_use = processed_df.drop(['isConfused'], axis=1)
        except KeyError:
            features_to_use = processed_df

        prediction = model.predict(features_to_use)
        total_predictions = len(prediction)
        confused_predictions = np.sum(prediction == 1)

        confusion_ratio = confused_predictions / total_predictions if total_predictions > 0 else 0.0
        is_user_confused = confusion_ratio > 0.3

        return {
            "Prediction": prediction.tolist(),
            "Confidence": model.predict_proba(features_to_use).tolist(),
            "ConfusionRatio": confusion_ratio,
            "UserConfused": is_user_confused
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Confusion Detector API. Use POST /predict to upload a CSV file."}
