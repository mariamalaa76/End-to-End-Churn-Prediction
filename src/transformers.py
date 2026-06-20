from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        eps = 1e-5
        
        if 'tenure' in X_copy.columns and 'monthlycharges' in X_copy.columns:
            X_copy['estimated_clv'] = X_copy['tenure'] * X_copy['monthlycharges']
            
        if 'totalcharges' in X_copy.columns and 'monthlycharges' in X_copy.columns:
            X_copy['charges_ratio'] = X_copy['totalcharges'] / (X_copy['monthlycharges'] + eps)

        if 'num_complaints' in X_copy.columns and 'num_service_calls' in X_copy.columns:
            X_copy['frustration_index'] = X_copy['num_complaints'] + X_copy['num_service_calls']
            
        if 'num_complaints' in X_copy.columns and 'tenure' in X_copy.columns:
            X_copy['complaints_per_month'] = X_copy['num_complaints'] / (X_copy['tenure'] + eps)

        if 'avg_monthly_gb' in X_copy.columns and 'num_services' in X_copy.columns:
            X_copy['gb_per_service'] = X_copy['avg_monthly_gb'] / (X_copy['num_services'] + eps)

        if 'signup_date' in X_copy.columns:
            signup_dt = pd.to_datetime(X_copy['signup_date'], errors='coerce')
            reference_date = pd.to_datetime('2026-06-20')  
            X_copy['account_age_days'] = (reference_date - signup_dt).dt.days
            X_copy['account_age_days'] = X_copy['account_age_days'].fillna(0)
            X_copy = X_copy.drop(columns=['signup_date'])

        return X_copy