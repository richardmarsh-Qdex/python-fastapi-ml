# FastAPI ML Model API

A machine learning model API built with FastAPI and scikit-learn.

## Features

- Model training endpoint
- Prediction endpoint
- Model management
- Health checks

## Installation

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## API Endpoints

- POST /train - Train a model
- POST /predict - Make predictions
- GET /model/info - Get model information
- DELETE /model - Delete model
