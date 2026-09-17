"""
modelling.py
Training model Machine Learning menggunakan MLflow autolog.

Author: gaji
Dataset: Wine Quality (Red Wine) - Binary Classification
Model: Random Forest Classifier
MLflow Tracking: localhost
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir: str) -> tuple:
    """Load preprocessed data dari folder."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model_with_autolog():
    """Train model menggunakan MLflow autolog."""
    
    # Path ke preprocessed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'wine_quality_preprocessing')
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("wine-quality-autolog")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="random_forest_autolog"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Print results
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"MODEL: Random Forest Classifier (Autolog)")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))
        
        print(f"\n✅ Model dan artefak berhasil disimpan ke MLflow!")
        print(f"   Tracking URI: http://127.0.0.1:5000")
        print(f"   Experiment: wine-quality-autolog")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_model_with_autolog()
