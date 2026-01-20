from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.sentiment_model import SentimentModel

router = APIRouter(prefix="/api/v1", tags=["predictions"])

class TextRequest(BaseModel):
    text: str

sentiment_model = SentimentModel()

@router.post("/sentiment")
def predict_sentiment(request: TextRequest):
    if not sentiment_model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    try:
        prediction = sentiment_model.predict(request.text)
        return {
            "text": request.text,
            "sentiment": "positive" if prediction == 1 else "negative",
            "prediction": int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
