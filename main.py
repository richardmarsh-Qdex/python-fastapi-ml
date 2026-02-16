from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

app = FastAPI(title="ML Model API", version="1.0.0")

# Simple model storage
MODEL_FILE = "model.pkl"

class PredictionRequest(BaseModel):
    features: list[float]

class ModelTrainRequest(BaseModel):
    data: list[list[float]]
    labels: list[int]

@app.get("/")
def read_root():
    return {"message": "ML Model API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/train")
def train_model(request: ModelTrainRequest):
    try:
        X = np.array(request.data)
        y = np.array(request.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        
        return {
            "message": "Model trained successfully",
            "accuracy": float(accuracy),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(request: PredictionRequest):
    if not os.path.exists(MODEL_FILE):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probabilities": probability,
            "features": request.features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
def model_info():
    if not os.path.exists(MODEL_FILE):
        return {"status": "no_model", "message": "Model not trained yet"}
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        return {
            "status": "trained",
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', None),
            "n_features": getattr(model, 'n_features_in_', None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/model")
def delete_model():
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        return {"message": "Model deleted successfully"}
    return {"message": "No model to delete"}
