# 🎓 Assignment Submission Delay Prediction System

A streamlined machine learning system to forecast student assignment submission delays 3 to 5 days before deadlines.

---

## 📁 Project Structure

```text
├── .github/workflows/ci.yml  # Automated CI/CD GitHub Actions pipeline
├── pytest.ini                # Pytest discovery configuration
├── train.py                  # Train ML model pipeline & save model.joblib
├── test.py                   # Automated tests for data, model, and API
├── main.py                   # FastAPI backend REST microservice
├── app.py                    # Streamlit interactive dashboard (Instructor & Student)
├── data.csv                  # Student submission dataset
├── model.joblib              # Trained XGBoost classification pipeline
├── requirements.txt          # Python dependencies
└── MiniProject-Guidelines.pdf # Project instructions
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Trains the XGBoost pipeline on `data.csv` and outputs evaluation metrics:
```bash
python train.py
```

### 3. Run Automated Tests
Verifies data schema, model accuracy, and API endpoints:
```bash
python test.py
# or with pytest:
pytest test.py
```

### 4. Start the FastAPI REST API
Runs the backend API at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`):
```bash
python main.py
# or
uvicorn main:app --reload
```

### 5. Launch the Streamlit Web Application
Runs the interactive dashboard at `http://localhost:8501`:
```bash
streamlit run app.py
```

---

## ⚙️ CI/CD Workflow (`.github/workflows/ci.yml`)

The repository includes an automated GitHub Actions pipeline triggered on `push`, `pull_request`, and `workflow_dispatch`:
1. **Multi-Version Testing:** Validates against Python 3.10 and Python 3.11 with pip caching.
2. **Linting & Code Quality:** Enforces syntax integrity using `flake8`.
3. **Training Verification:** Automatically executes `train.py` to confirm model pipeline stability.
4. **Automated Testing:** Runs `pytest -v` across data integrity, model metrics, and FastAPI endpoints.
5. **Smoke Import Test:** Verifies FastAPI and Streamlit runtime imports.
6. **Model Artifact Upload:** Automatically archives `model.joblib` as a downloadable GitHub Actions artifact.
