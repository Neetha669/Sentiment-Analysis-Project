import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers 

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load Dataset

df = pd.read_csv("reviews.csv")

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(label_map)

# Using same data for demo
train_df = df.copy()
test_df = df.copy()


# Tokenizer

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        padding=True
    )


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)


# Load Model

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# Accuracy Function

def compute_metrics(pred):

    predictions = pred.predictions.argmax(-1)

    accuracy = accuracy_score(pred.label_ids, predictions)

    return {
        "accuracy": accuracy
    }
# Training Arguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2
)

# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# MLflow

mlflow.set_experiment("Sentiment_Analysis_DistilBERT")

with mlflow.start_run():

    # Train Model
    trainer.train()

    # Evaluate
    result = trainer.evaluate()

    # Predictions
    pred = trainer.predict(test_dataset)

    predictions = pred.predictions.argmax(axis=1)

    accuracy = accuracy_score(
        pred.label_ids,
        predictions
    )

    print("Accuracy:", accuracy)

    # Classification Report
    report = classification_report(
        pred.label_ids,
        predictions,
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(
        pred.label_ids,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    disp.plot(cmap="Blues")

    plt.savefig("confusion_matrix.png")

    plt.close()

    # Save Model
    trainer.save_model("model")
    tokenizer.save_pretrained("model")

    # MLflow Parameters
    mlflow.log_param("Model", "DistilBERT")
    mlflow.log_param("Epochs", training_args.num_train_epochs)
    mlflow.log_param("Batch Size", training_args.per_device_train_batch_size)
    mlflow.log_param("Learning Rate", training_args.learning_rate)

    # MLflow Metrics

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Eval Loss", result["eval_loss"])

    # Register Model
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    mlflow.transformers.log_model(
        transformers_model=sentiment_pipeline,
        artifact_path="SentimentModel",
        registered_model_name="SentimentAnalysisModel"
    )

    # Log Artifacts
    mlflow.log_artifact("classification_report.txt")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("reviews.csv")
    mlflow.log_artifact("Dockerfile")
    mlflow.log_artifact("model_card.md")

print("Training Completed Successfully")