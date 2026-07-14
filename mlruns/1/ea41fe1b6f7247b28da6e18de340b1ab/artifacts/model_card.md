# Model Card

## Model Name

DistilBERT Sentiment Analysis

## Model Type

Text Classification

## Framework

Hugging Face Transformers

## Dataset

Customer Reviews Dataset

## Labels

- Positive
- Negative
- Neutral

## Input

Customer review text

## Output

Predicted sentiment

## Training

Fine-tuned DistilBERT model using PyTorch and Hugging Face Trainer.

## Metrics

- Accuracy
- Precision
- Recall
- F1-score

## Limitations

- Works only for English reviews.
- Small datasets reduce prediction accuracy.
- May not correctly detect sarcasm.