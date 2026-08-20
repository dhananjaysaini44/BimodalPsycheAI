from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.orchestrator.sentiment_scorer import score_turn_sentiment
from app.orchestrator.crisis_detector import check_crisis_signal

router = APIRouter()


class TextPayload(BaseModel):
    text: str


@router.post("/sentiment")
def sentiment_route(payload: TextPayload):
    try:
        score = score_turn_sentiment(payload.text)
        return {"sentiment_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crisis")
def crisis_route(payload: TextPayload):
    try:
        is_crisis = check_crisis_signal(payload.text)
        return {"is_crisis": is_crisis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
