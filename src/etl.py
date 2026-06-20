import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.transformers import TelecomFeatureEngineer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_complete_pipeline(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop' 
    )

    full_pipeline = Pipeline(steps=[
        ('feature_engineering', TelecomFeatureEngineer()),
        ('preprocessing', preprocessor)
    ])
    
    return full_pipeline

def run_etl():
    print("=== Starting Professional ETL Pipeline ===")
    config = load_config()
    
    df = pd.read_csv(config['data']['raw_path'])
    print(f"Original Data Shape: {df.shape}")
    
    target_col = config['data']['target_column']
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split successfully into Train (80%) and Test (20%).")
    
    num_feats = config['features']['numerical_features']
    cat_feats = config['features']['categorical_features']
    
    pipeline = build_complete_pipeline(num_feats, cat_feats)
    
    print("Fitting the Pipeline on Training Data...")
    X_train_processed = pipeline.fit_transform(X_train)
    
    print("Transforming Testing Data using learned values...")
    X_test_processed = pipeline.transform(X_test)
    
    cat_encoder = pipeline.named_steps['preprocessing'].named_transformers_['cat'].named_steps['onehot']
    encoded_cat_feats = list(cat_encoder.get_feature_names_out(cat_feats))
    all_features_names = num_feats + encoded_cat_feats
    
    processed_folder = config['data']['processed_folder']
    os.makedirs(processed_folder, exist_ok=True)
    
    train_processed_df = pd.DataFrame(X_train_processed, columns=all_features_names)
    train_processed_df[target_col] = y_train.values
    
    test_processed_df = pd.DataFrame(X_test_processed, columns=all_features_names)
    test_processed_df[target_col] = y_test.values
    
    train_processed_df.to_csv(os.path.join(processed_folder, "train_processed.csv"), index=False)
    test_processed_df.to_csv(os.path.join(processed_folder, "test_processed.csv"), index=False)
    
    joblib.dump(pipeline, os.path.join(processed_folder, "full_pipeline.joblib"))
    
    print(f"=== ETL Pipeline Executed Successfully! ===")
    print(f"Saved processed data and pipeline object to: {processed_folder}")

if __name__ == "__main__":
    run_etl()