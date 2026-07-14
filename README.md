# Sentiment Analysis and Review Intelligence using DistilBERT

## 📌 Project Overview

Sentiment Analysis and Review Intelligence is an NLP-based project that classifies user reviews into **Positive, Negative, and Neutral sentiments** using a fine-tuned **Hugging Face DistilBERT Transformer model**.

The project provides a real-time sentiment prediction API using **FastAPI**, tracks experiments using **MLflow**, uses **Docker for containerization**, and is deployed on **Render Cloud Platform**.

---

## 🚀 Features

- Automated sentiment classification from text reviews
- Fine-tuned DistilBERT model for NLP classification
- Real-time prediction through REST API
- MLflow experiment tracking and model evaluation
- Docker-based deployment
- Model Card documentation
- Cloud deployment using Render

---

## 🛠️ Tech Stack

**Machine Learning & NLP**
- Python
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Scikit-learn

**Backend & Deployment**
- FastAPI
- Docker
- Render

**Experiment Tracking**
- MLflow

**Version Control**
- Git & GitHub

---

## 📂 Project Structure

```text
│
├── __pycache__/                  - Python cache files
│
├── .vscode/                      - VS Code configuration files
│
├── mlruns/                       - MLflow experiment tracking data
│
├── model/                        - Fine-tuned DistilBERT model files
│
├── myenv/                        - Python virtual environment
│
├── .dockerignore                 - Files ignored during Docker build
│
├── .gitattributes                - Git attributes configuration
│
├── .gitignore                    - Git ignored files
│
├── app.py                        - FastAPI application and API endpoints
│
├── classification_report.txt     - Model classification performance report
│
├── confusion_matrix.png          - Confusion matrix visualization
│
├── Dockerfile                    - Docker container configuration
│
├── IS Project                    - Project related file/document
│
├── mlflow.db                     - MLflow database for experiment tracking
│
├── model_card.md                 - Model documentation
│
├── prediction.py                 - Model loading and sentiment prediction logic
│
├── README.md                     - Project documentation
│
├── requirements.txt              - Required Python dependencies
│
├── reviews.csv                   - Sentiment analysis dataset
│
└── train.py                      - Model training and fine-tuning script
```
## 🔄 Project Workflow

1. Data collection and preprocessing
2. Fine-tuning the Hugging Face DistilBERT model for sentiment classification
3. Model evaluation using accuracy, precision, recall, and F1-score
4. Tracking experiments, parameters, and metrics using MLflow
5. Developing a REST API using FastAPI
6. Containerizing the application using Docker
7. Deploying the application on Render Cloud Platform

---

## 🤖 Model Details

**Model:** Fine-tuned Hugging Face DistilBERT  
**Task:** Sentiment Classification  
**Framework:** Hugging Face Transformers and PyTorch  

The model classifies text reviews into three sentiment categories:

- Negative
- Neutral
- Positive

---

## API Implementation

The FastAPI application provides REST API endpoints for real-time sentiment prediction.

**API Endpoints**

**1. Health Check Endpoint**


GET /

Checks whether the API service is running successfully.


**2. Prediction Endpoint**


POST /predict

Receives text input and returns the predicted sentiment classification.

---


## 📊 MLflow Experiment Tracking

MLflow is used for tracking and managing machine learning experiments.

It tracks:

- Model parameters
- Training metrics
- Evaluation results
- Experiment runs
- Model performance

---

## 🐳 Docker Deployment

Docker is used to containerize the FastAPI application and provide a consistent deployment environment.

Docker helps in:

- Creating application containers
- Managing dependencies
- Running the application across different environments

---

## ☁️ Deployment

The application is deployed using **Render Cloud Platform**.

Deployment process:

1. Push the project code to GitHub repository
2. Connect GitHub repository with Render
3. Configure deployment settings
4. Build and deploy the application

---

## 📦 Deliverables

- Sentiment Classification Model
- FastAPI REST API
- Dockerized Application
- MLflow Experiment Tracking
- Model Card Documentation
- GitHub Repository
- Render Cloud Deployment

---

# Sentiment Analysis Project Setup & Execution Commands

1. Create Virtual Environment
```text
python -m venv myenv
```

2. Activate Virtual Environment (Windows)
```text
myenv\Scripts\activate
```

3. Install Required Libraries
```text
pip install -r requirements.txt
```

4. Train the DistilBERT Sentiment Analysis Model
```text
python train.py
```


5. Run MLflow Tracking UI
```text
mlflow ui
```

Open in Browser:
```text
http://localhost:5000
```

6. Run FastAPI Application
```text
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open API Documentation:
```text
http://localhost:8000/docs
```

7. Build Docker Image
```text
docker build -t sentiment-analysis .
```

8. Run Docker Container
```text
docker run -p 8000:8000 sentiment-analysis
```

# 9. GitHub Commands

Initialize Git Repository
```text
git init
```
Add Project Files
```text
git add .
```
Commit Changes
```text
git commit -m "Sentiment Analysis project with DistilBERT and FastAPI"
```
Connect GitHub Repository
```text
git remote add origin <your-github-repository-url>
```
Rename Branch and Push Code
```text
git branch -M main
git push -u origin main
```
## 10. Render Deployment Steps


1. Push project code to GitHub
2. Login to Render
3. Create New Web Service
4. Connect GitHub Repository
5. Select Docker as Environment
6. Deploy Application


## Project Structure Used:
app.py          -> FastAPI API implementation

prediction.py   -> Model prediction logic

train.py        -> DistilBERT model training

model/          -> Saved trained model files

mlruns/         -> MLflow experiment tracking

Dockerfile      -> Docker deployment configuration

requirements.txt -> Required Python packages

## 👩‍💻 Author

**Neetha**  
B.E Computer Science & Engineering (Data Science)
