# Sentiment Analysis and Review Intelligence

## Project Description

This project classifies customer reviews into Positive, Negative, and Neutral sentiments using Hugging Face DistilBERT.

## Technologies Used

- Python
- Hugging Face DistilBERT
- PyTorch
- FastAPI
- Docker
- AWS S3
- AWS Elastic Beanstalk

## Project Structure

```
sentiment_analysis/
│── app.py
│── train.py
│── prediction.py
│── reviews.csv
│── requirements.txt
│── Dockerfile
│── model/
│── results/
```

## Installation

```bash
git clone <your-repository-url>
cd sentiment_analysis
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## API

### POST /predict

Input

```json
{
  "text":"This product is amazing"
}
```

Output

```json
{
  "review":"This product is amazing",
  "sentiment":"Positive"
}
```

## Author

Neetha