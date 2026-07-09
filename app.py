from fastapi import FastAPI
from pydantic import BaseModel
from prediction import predict_sentiment

app=FastAPI()

class Review(BaseModel):
    text:str

@app.get("/")
def home():
    return {"message":"Sentiment Analysis API"}

@app.post("/predict")
def predict(review:Review):
    result=predict_sentiment(review.text)
    return {
        "review":review.text,
        "sentiment":result

    }