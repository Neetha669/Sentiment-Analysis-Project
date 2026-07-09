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
evaluation_strategy="epoch",
per_device_train_batch_size=8,
num_train_epochs=2
)

trainer=Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset
)

trainer.train()
model.save_pretrained("model")
tokenizer.save_pretrained("model")