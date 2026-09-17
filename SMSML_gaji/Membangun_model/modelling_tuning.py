"""
modelling_tuning.py
Training model ML dengan Hyperparameter Tuning dan Manual Logging ke MLflow.
Level: Advanced - dengan DagsHub integration dan artefak tambahan.

Author: gaji
Dataset: Breast Cancer Wisconsin - Binary Classification
Model: Random Forest Classifier + GridSearchCV
MLflow: Manual logging (bukan autolog)
Artefak tambahan: confusion_matrix, feature_importance, classification_report, ROC curve
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import tempfile
import warnings
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

def load_preprocessed_data(data_dir: str) -> tuple:
    """Load preprocessed data dari folder."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def create_confusion_matrix_plot(y_test, y_pred, save_path):
    """Membuat dan menyimpan confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title('Confusion Matrix - Tuned Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def create_feature_importance_plot(model, feature_names, save_path):
    """Membuat dan menyimpan feature importance plot."""
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title('Feature Importance - Tuned Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance saved: {save_path}")


def create_roc_curve_plot(y_test, y_prob, save_path):
    """Membuat dan menyimpan ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Tuned Random Forest', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved: {save_path}")


def create_prediction_distribution_plot(y_prob, save_path):
    """Membuat dan menyimpan prediction probability distribution plot."""
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.xlabel('Probability of Benign')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Prediction distribution saved: {save_path}")


def train_model_with_tuning():
    """Train model dengan hyperparameter tuning dan manual logging."""
    
    # Path ke preprocessed data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'breast_cancer_preprocessing')
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("breast-cancer-tuning")
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("="*60)
    print("HYPERPARAMETER TUNING - Random Forest Classifier")
    print("="*60)
    print(f"\nParameter Grid:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    
    # GridSearchCV
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print("\nMemulai GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score (F1): {grid_search.best_score_:.4f}")
    
    # Predict
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'best_cv_score': grid_search.best_score_,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[1]
    }
    
    # Manual logging ke MLflow
    with mlflow.start_run(run_name="random_forest_tuned_advanced"):
        
        # === LOG PARAMETERS ===
        print("\n📝 Logging parameters...")
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # === LOG METRICS ===
        print("📊 Logging metrics...")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # === LOG MODEL ===
        print("🤖 Logging model...")
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="breast-cancer-rf-tuned"
        )
        
        # === LOG ARTEFAK TAMBAHAN (Advanced: autolog + minimal 2 artefak tambahan) ===
        print("\n📦 Logging artefak tambahan...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Artefak 1: Confusion Matrix Plot
            cm_path = os.path.join(tmp_dir, 'confusion_matrix.png')
            create_confusion_matrix_plot(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path, "plots")
            
            # Artefak 2: Feature Importance Plot
            fi_path = os.path.join(tmp_dir, 'feature_importance.png')
            create_feature_importance_plot(best_model, X_train.columns.tolist(), fi_path)
            mlflow.log_artifact(fi_path, "plots")
            
            # Artefak 3: ROC Curve Plot
            roc_path = os.path.join(tmp_dir, 'roc_curve.png')
            create_roc_curve_plot(y_test, y_prob, roc_path)
            mlflow.log_artifact(roc_path, "plots")
            
            # Artefak 4: Prediction Distribution Plot
            pred_dist_path = os.path.join(tmp_dir, 'prediction_distribution.png')
            create_prediction_distribution_plot(y_prob, pred_dist_path)
            mlflow.log_artifact(pred_dist_path, "plots")
            
            # Artefak 5: Classification Report (text)
            report_path = os.path.join(tmp_dir, 'classification_report.txt')
            report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
            with open(report_path, 'w') as f:
                f.write("Classification Report - Tuned Random Forest\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            mlflow.log_artifact(report_path, "reports")
            print(f"  ✓ Classification report saved")
            
            # Artefak 6: Grid Search Results (CSV)
            gs_results_path = os.path.join(tmp_dir, 'grid_search_results.csv')
            gs_results = pd.DataFrame(grid_search.cv_results_)
            gs_results.to_csv(gs_results_path, index=False)
            mlflow.log_artifact(gs_results_path, "reports")
            print(f"  ✓ Grid search results saved")
            
            # Artefak 7: Model metadata (JSON)
            metadata_path = os.path.join(tmp_dir, 'model_metadata.json')
            metadata = {
                "model_type": "RandomForestClassifier",
                "best_params": best_params,
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "features": X_train.columns.tolist(),
                "target": "target",
                "dataset": "Breast Cancer (Red)",
                "preprocessing": "StandardScaler + IQR Outlier Clipping"
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(metadata_path, "reports")
            print(f"  ✓ Model metadata saved")
        
        # === TAGS ===
        mlflow.set_tag("author", "gaji")
        mlflow.set_tag("dataset", "breast-cancer-wisconsin")
        mlflow.set_tag("level", "advanced")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        
        run_id = mlflow.active_run().info.run_id
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING SELESAI")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nBest Parameters: {best_params}")
    print(f"\n✅ Semua artefak berhasil disimpan ke MLflow!")
    print(f"   Tracking URI: http://127.0.0.1:5000")
    print(f"   Experiment: breast-cancer-tuning")
    print(f"   Run ID: {run_id}")
    print(f"\n📦 Artefak yang di-log:")
    print(f"   1. Model (sklearn)")
    print(f"   2. Confusion Matrix (plot)")
    print(f"   3. Feature Importance (plot)")
    print(f"   4. ROC Curve (plot)")
    print(f"   5. Prediction Distribution (plot)")
    print(f"   6. Classification Report (text)")
    print(f"   7. Grid Search Results (CSV)")
    print(f"   8. Model Metadata (JSON)")


if __name__ == "__main__":
    train_model_with_tuning()
