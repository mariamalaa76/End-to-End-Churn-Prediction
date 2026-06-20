تفضلي، هذا هو الكود بالكامل داخل بلوك برمجى واحد (Markdown Code Block)؛ لتتمكني من نسخه بضغطة زر واحدة (Copy) ووضعه مباشرة في ملف الـ `README.md`:

```markdown
# 📱 Telecom Churn Prediction: End-to-End Enterprise MLOps Pipeline

An enterprise-grade, production-ready MLOps pipeline designed to predict telecom customer churn using **XGBoost**. This project transitions from experimental notebooks to a fully modular, decoupled architecture featuring configuration-driven workflows, experiment tracking, Docker containerization, and automated CI/CD.

---

## 🏗️ System Architecture & Workflow

The project is structured following professional software engineering principles, decoupling data ingestion, feature engineering, training, and deployment.

```text
├── .github/workflows/   # CI/CD: GitHub Actions for linting and deployment
├── app/                 # Production API (FastAPI)
│   └── main.py
├── data/                # Data storage (Raw & Processed)
│   ├── raw/
│   └── processed/
├── models/              # Saved serialization of the best model (.joblib)
├── src/                 # Modular ML Pipeline
│   ├── __init__.py
│   ├── etl.py          # Extraction, Transformation, and Data Splitting
│   ├── train.py        # Advanced Cloud Hyperparameter Tuning & GridSearch
│   └── pipeline.py     # Main coordinator for local execution
├── config.yaml          # Single source of truth for paths & parameters
├── Dockerfile           # Application containerization configuration
└── requirements.txt     # Locked dependencies

```

---

## 🚀 Key MLOps Features

### 1. Advanced Cloud Hyperparameter Tuning (GridSearch)

To handle a large-scale dataset (~1M records) without local hardware bottlenecks, the training pipeline was offloaded to high-performance cloud environments using **Stratified 2-Fold Cross-Validation** to combat overfitting.

* **Hyperparameter Grid Explored:** `max_depth` [6, 8, 10], `learning_rate` [0.05, 0.1, 0.15], and balanced class weights.
* **The Challenge:** Extreme class imbalance (~9:1 ratio).
* **The Solution:** Optimized the objective function using dynamic `scale_pos_weight` computation to find the optimal decision boundary between **Precision** and **Recall**.

### 2. Experiment Tracking with MLflow

All training runs, performance metrics, and artifact generations are fully tracked using **MLflow Autologging**. This provides a visual dashboard to audit model history, compare metrics like F1-Score and ROC-AUC, and ensure complete reproducibility.

### 3. Production API & Dockerization

* **FastAPI Backend:** Built a high-performance web service capable of serving real-time customer data predictions with sub-second latency.
* **Docker Containerization:** Isolated the entire runtime environment (Python 3.11, system packages, and dependencies) into a standalone Docker Image, eliminating configuration drifts between staging and production.

### 4. Continuous Integration (CI/CD) via GitHub Actions

An automated pipeline triggers on every `push` to the `main` branch:

* **Code Quality Assurance:** Automates syntax and style checking using `flake8` to prevent technical debt.
* **Build Verification:** Simulates an isolated environment setup to guarantee that the production pipeline runs seamlessly without regressions.

---

## 📊 Evaluation Metrics (Production Ready Baseline)

Following intensive grid-search optimization, the final model achieved highly stable deployment metrics:

* **Overall Global Accuracy:** `86%`
* **Class 0 (Retained Customers) Recall:** `92%` (Minimizes false-positive retention budget spending)
* **Class 1 (Churned Customers) Precision:** Elevated to `25%` to significantly reduce false alarms while maintaining predictable capture rates.
* **ROC-AUC Score:** `0.6844`

---

## 🛠️ Local Setup & Execution

### Prerequisites

Make sure you have Python 3.11+ and Docker installed locally.

### 1. Installation & Environment Setup

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/mariamalaa76/End-to-End-Churn-Prediction.git](https://github.com/mariamalaa76/End-to-End-Churn-Prediction.git)
cd End-to-End-Churn-Prediction
pip install -r requirements.txt

```

### 2. Run the End-to-End Pipeline

Execute the modular pipeline from raw data extraction to model registration:

```bash
python -m src.pipeline

```

### 3. Launch MLflow Dashboard

Inspect the logs, plots, and parameters in your browser:

```bash
mlflow ui

```

*Navigate to `http://127.0.0.1:5000*`

### 4. Containerized Deployment (Docker)

Build and spin up the production API locally:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api

```

*Access the interactive API documentation at `http://127.0.0.1:8000/docs*`

```

```
