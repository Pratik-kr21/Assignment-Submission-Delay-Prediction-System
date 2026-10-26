# 🎓 EduPredict-MLOps: Assignment Submission Delay Prediction System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io/)

An end-to-end Machine Learning Operations (MLOps) platform that forecasts student assignment submission delays 3 to 5 days before deadlines. Designed for educational institutions and Learning Management Systems (Canvas, Moodle, Blackboard) to empower instructors with automated early-warning alerts and pedagogical interventions.

---

## 🌟 Key Features

* **Proactive Behavioral Forecasting**: Evaluates student clickstream velocity, historical submission punctuality (RAST), and concurrent deadline loads to compute delay probabilities before deadlines occur.
* **MLflow Experiment Tracking & Registry**: Automatic tracking of hyperparameters, classification metrics (F1-score, PR-AUC, ROC-AUC), and model artifacts.
* **Production REST API**: High-performance FastAPI backend providing `/predict` (single student with risk tier) and `/batch_predict` (cohort screening).
* **Interactive Streamlit Web Dashboard**:
  * **Instructor Portal**: Cohort risk heatmaps, student sorting by delay risk, and 1-click automated intervention email drafting.
  * **Student Self-Check Portal**: Self-service submission timeline evaluation and study planning suggestions.
  * **MLOps Observability Portal**: Real-time feature drift detection (Kolmogorov-Smirnov statistical testing) across semester transitions.
* **Containerized Architecture**: Multi-container setup orchestrating the FastAPI server, Streamlit UI, and MLflow tracking server via Docker Compose.
* **Continuous Integration**: Unit tests covering feature transformations and API contracts using `pytest`.

---

## 🏗️ Architecture Blueprint

```text
[LMS / OULAD Data] ──> [DVC Data Versioning] ──> [Feature Pipeline (RAST + Clicks)]
                                                            │
                                                            ▼
[Evidently Drift Monitor] <── [FastAPI Microservice] <── [MLflow Registry / XGBoost]
          │                           │
          ▼                           ▼
[Retrain Alert Trigger]     [Streamlit UI Dashboard]
```

---

## 📁 Repository Structure

```text
├── .gitignore               # Ignores venvs, data binaries, MLflow runs, checkpoints
├── Dockerfile               # Multi-stage production container image
├── docker-compose.yml       # Orchestrates API, Streamlit, and MLflow services
├── requirements.txt         # Project dependencies
├── config/
│   └── config.yaml          # Centralized configuration (parameters, paths, features)
├── data/
│   ├── raw/                 # Raw datasets (tracked via DVC)
│   └── processed/           # Transformed feature sets (tracked via DVC)
├── docs/
│   └── Phase1_2_Project_Proposal_and_System_Design.md  # Official Phase 1-2 submission deliverable
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py  # Data loading and OULAD-mirror synthetic generator
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py# ColumnTransformer preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py   # MLflow training pipeline (XGBoost, RF, LogReg)
│   │   └── predict_model.py # Inference, risk tiering, and action engine
│   └── monitoring/
│       ├── __init__.py
│       └── drift_detector.py# Two-sample KS-test data drift detector
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI microservice
│   └── schemas.py           # Pydantic request/response schemas
├── frontend/
│   └── app.py               # Streamlit web dashboard
└── tests/
    ├── __init__.py
    ├── test_features.py     # Unit tests for feature extraction
    └── test_api.py          # Integration tests for FastAPI endpoints
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
* Python 3.10+ (Python 3.11 recommended)
* Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Pratik-kr21/Assignment-Submission-Delay-Prediction-System.git
cd Assignment-Submission-Delay-Prediction-System

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Training with MLflow
Train the baseline and XGBoost models. This will log metrics and serialize the model artifact to `models/model.joblib`:
```bash
python -m src.models.train_model
```

To view the MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### 4. Run the Application

#### Option A: Run Services Locally
In Terminal 1 (Start FastAPI backend):
```bash
uvicorn api.main:app --reload --port 8000
# API docs available at: http://localhost:8000/docs
```

In Terminal 2 (Start Streamlit UI):
```bash
streamlit run frontend/app.py --server.port 8501
# Dashboard available at: http://localhost:8501
```

#### Option B: Run via Docker Compose
```bash
docker-compose up --build
```
* **FastAPI Service**: `http://localhost:8000/docs`
* **Streamlit Dashboard**: `http://localhost:8501`
* **MLflow Tracking**: `http://localhost:5000`

---

## 🧪 Running Unit Tests

Run the complete test suite with `pytest`:
```bash
pytest -v
```

---

## 📄 Submission Documents
* **Phase 1–2 Deliverable**: [docs/Phase1_2_Project_Proposal_and_System_Design.md](docs/Phase1_2_Project_Proposal_and_System_Design.md)
* **Course Guidelines Reference**: [MiniProject-Guidelines.pdf](MiniProject-Guidelines.pdf)

---

## 👥 Team & Responsibilities (MLOps Mini Project)
* **Member 1 (Data & Feature Engineer)**: Data ingestion, DVC setup, feature engineering pipeline.
* **Member 2 (ML & MLOps Engineer)**: Model training, MLflow tracking, hyperparameter tuning, testing.
* **Member 3 (Deployment & UI Engineer)**: FastAPI REST endpoints, Streamlit dashboard, Docker, drift monitoring.
