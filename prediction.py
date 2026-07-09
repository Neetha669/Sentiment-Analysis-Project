from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import torch

tokenizer=DistilBertTokenizerFast.from_pretrained("model")
model=DistilBertForSequenceClassification.from_pretrained("model")

labels={
0:"Negative",
1:"Neutral",
2:"Positive"
}

def predict_sentiment(text):
    inputs=tokenizer(text,return_tensors="pt",truncation=True,padding=True)

    outputs=model(**inputs)
    prediction=torch.argmax(outputs.logits,dim=1).item()
    return labels[prediction]