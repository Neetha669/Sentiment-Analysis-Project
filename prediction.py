from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return labels[prediction]