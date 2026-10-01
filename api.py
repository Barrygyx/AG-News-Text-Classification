from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict_news


app = FastAPI(
    title="AG News Classification API",
    description="Classify news articles into World, Sports, Business, or Sci/Tech.",
    version="1.0"
)


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
def predict(request: NewsRequest):
    category = predict_news(request.text)

    return {
        "text": request.text,
        "prediction": category
    }