import mlflow
import mlflow.pytorch

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

df = pd.read_csv("reviews.csv")

label_map={
    "negative":0,
    "neutral":1,
    "positive":2
}

df["label"]=df["sentiment"].map(label_map)
train_df,test_df=train_test_split(df,test_size=0.2)
tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_dataset=Dataset.from_pandas(train_df)
test_dataset=Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(batch["review"],truncation=True,padding=True)

train_dataset=train_dataset.map(tokenize,batched=True)
test_dataset=test_dataset.map(tokenize,batched=True)

train_dataset.set_format("torch",
columns=["input_ids","attention_mask","label"])

test_dataset.set_format("torch",
columns=["input_ids","attention_mask","label"])

model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=3)

training_args=TrainingArguments(
output_dir="./results",
eval_strategy="epoch",
per_device_train_batch_size=8,
num_train_epochs=2
)

trainer=Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset
)
mlflow.set_experiment("Sentiment_Analysis_DistilBERT")

with mlflow.start_run():

    trainer.train()
    mlflow.log_param("Model", "DistilBERT")
    mlflow.log_param("Epochs", training_args.num_train_epochs)
    mlflow.log_param("Batch Size", training_args.per_device_train_batch_size)
    mlflow.log_param("Learning Rate", training_args.learning_rate)
        
    result = trainer.evaluate()

    mlflow.log_metric("Eval Loss", result["eval_loss"])

    if "eval_accuracy" in result:
        mlflow.log_metric("Accuracy", result["eval_accuracy"])
        
        mlflow.pytorch.log_model(model, "model")
        mlflow.pytorch.log_model(tokenizer, "tokenizer")
        
        trainer.save_model("model")

        mlflow.pytorch.log_model(model, "SentimentModel")

        mlflow.log_artifact("reviews.csv")
        mlflow.log_artifact("Dockerfile")
        mlflow.log_artifact("model_card.md")