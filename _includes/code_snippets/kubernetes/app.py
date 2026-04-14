from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # In practice you'd load a real model here — joblib, torch, etc.
    score = sum(request.features) / len(request.features)
    return PredictResponse(score=score)
