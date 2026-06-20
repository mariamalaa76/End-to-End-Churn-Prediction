import os
import yaml
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.xgboost

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_class_weights(y_train):
    negative_cases = np.sum(y_train == 0)
    positive_cases = np.sum(y_train == 1)
    scale_pos_weight = negative_cases / positive_cases
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
    return scale_pos_weight

def train_model():
    print("=== Starting Training Pipeline with MLflow ===")
    config = load_config()
    processed_folder = config['data']['processed_folder']
    target_col = config['data']['target_column']
    
    train_path = os.path.join(processed_folder, "train_processed.csv")
    test_path = os.path.join(processed_folder, "test_processed.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("Processed training data not found. Please run the ETL pipeline first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_experiment("Telecom_Churn_Production")
    pos_weight = calculate_class_weights(y_train)
    
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": pos_weight,  
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    with mlflow.start_run(run_name="XGBoost_Base_Model"):
        print("Training XGBoost Classifier...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("\nModel Evaluation Results:")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        mlflow.xgboost.log_model(model, artifact_path="model")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "xgboost_model.joblib"))
        print(f"Model artifact saved locally to: {model_dir}/xgboost_model.joblib")
        
    print("=== Training Pipeline Finished & Logged to MLflow! ===")

if __name__ == "__main__":
    train_model()