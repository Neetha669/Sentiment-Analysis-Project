import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("reviews.csv")

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(label_map)

train_df = df.copy()
test_df = df.copy()

# ==========================
# Tiny BERT Model
# ==========================

MODEL_NAME = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        padding=True,
        max_length=128
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

# ==========================
# Load Model
# ==========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# ==========================
# Metrics
# ==========================

def compute_metrics(pred):

    predictions = pred.predictions.argmax(-1)

    accuracy = accuracy_score(
        pred.label_ids,
        predictions
    )

    return {
        "accuracy": accuracy
    }

# ==========================
# Training Arguments
# ==========================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2
)

# ==========================
# Trainer
# ==========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ==========================
# MLflow
# ==========================

mlflow.set_experiment("Sentiment_Analysis_BERT_TINY")

with mlflow.start_run():

    trainer.train()

    result = trainer.evaluate()

    pred = trainer.predict(test_dataset)

    predictions = pred.predictions.argmax(axis=1)

    accuracy = accuracy_score(
        pred.label_ids,
        predictions
    )

    print("Accuracy:", accuracy)

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

    trainer.save_model("model")

    tokenizer.save_pretrained("model")

    mlflow.log_param("Model", MODEL_NAME)
    mlflow.log_param("Epochs", training_args.num_train_epochs)
    mlflow.log_param("Batch Size", training_args.per_device_train_batch_size)
    mlflow.log_param("Learning Rate", training_args.learning_rate)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Eval Loss", result["eval_loss"])

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    mlflow.transformers.log_model(
        transformers_model=sentiment_pipeline,
        artifact_path="SentimentModel"
    )

    mlflow.log_artifact("classification_report.txt")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("reviews.csv")
    mlflow.log_artifact("Dockerfile")
    mlflow.log_artifact("model_card.md")

print("Training Completed Successfully")