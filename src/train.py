import os
import yaml
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    return scale_pos_weight

def train_model():
    print("=== Starting Advanced Optimization Pipeline with MLflow ===")
    config = load_config()
    processed_folder = config['data']['processed_folder']
    target_col = config['data']['target_column']
    
    train_path = os.path.join(processed_folder, "train_processed.csv")
    test_path = os.path.join(processed_folder, "test_processed.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("Processed data not found. Please run ETL first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_experiment("Telecom_Churn_Production")
    
    pos_weight = calculate_class_weights(y_train)
    
    # التعديل الاحترافي: وسعنا الـ depth وجربنا وزن متزن للـ Classes لرفع الـ Precision
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 150],
        'scale_pos_weight': [pos_weight, pos_weight * 0.5]  
    }
    
    # تقليل الـ splits لـ 2 لتسريع عملية البحث حسابياً مع الداتا الضخمة
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    mlflow.xgboost.autolog()
    
    with mlflow.start_run(run_name="XGBoost_GridSearch_Optimization"):
        print("Running Grid Search Cross-Validation...")
        base_xgb = XGBClassifier(eval_metric="logloss", random_state=42)
        
        # هنقيس الأداء بالـ roc_auc لأنه يعبر عن قوة الفصل العامة للموديل
        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"\n🏆 Best Parameters Found: {grid_search.best_params_}")
        
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("\n Best Model Evaluation Results:")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        mlflow.log_metric("final_test_f1", f1)
        mlflow.log_metric("final_test_roc_auc", roc_auc)
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(model_dir, "xgboost_model.joblib"))
        print(f"Best model saved to: {model_dir}/xgboost_model.joblib")
        
    print("=== Optimization Pipeline Finished & Logged to MLflow! ===")

if __name__ == "__main__":
    train_model()