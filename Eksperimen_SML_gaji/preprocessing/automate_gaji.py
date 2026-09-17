"""
automate_gaji.py
Script otomatis untuk preprocessing dataset Wine Quality.
Mengkonversi langkah-langkah eksperimen dari notebook ke pipeline otomatis.

Author: gaji
Dataset: Wine Quality (Red Wine)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV.
    
    Args:
        filepath: Path ke file CSV dataset
        
    Returns:
        DataFrame dengan data yang sudah di-load
    """
    logger.info(f"Loading data dari: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    
    df = pd.read_csv(filepath, sep=';')
    logger.info(f"Data berhasil di-load. Shape: {df.shape}")
    logger.info(f"Kolom: {list(df.columns)}")
    
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Melakukan Exploratory Data Analysis (EDA) dasar.
    
    Args:
        df: DataFrame input
        
    Returns:
        Dictionary berisi hasil EDA
    """
    logger.info("Memulai EDA...")
    
    eda_results = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "statistics": {}
    }
    
    # Statistik deskriptif
    stats = df.describe()
    eda_results["statistics"] = stats.to_dict()
    
    # Distribusi target
    if 'quality' in df.columns:
        quality_dist = df['quality'].value_counts().to_dict()
        eda_results["quality_distribution"] = {str(k): int(v) for k, v in quality_dist.items()}
        logger.info(f"Distribusi Quality: {quality_dist}")
    
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplikat: {eda_results['duplicates']}")
    
    return eda_results


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values pada dataset.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa missing values
    """
    logger.info("Menangani missing values...")
    
    missing_before = df.isnull().sum().sum()
    
    # Isi missing values numerik dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled with median ({median_val})")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa duplikat
    """
    duplicates = df.duplicated().sum()
    logger.info(f"Menghapus {duplicates} baris duplikat...")
    
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Shape setelah hapus duplikat: {df.shape}")
    
    return df


def handle_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr') -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR.
    
    Args:
        df: DataFrame input
        columns: List kolom untuk deteksi outlier (default: semua numerik)
        method: Metode deteksi ('iqr')
        
    Returns:
        DataFrame dengan outlier yang sudah ditangani
    """
    logger.info(f"Menangani outlier dengan metode {method}...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target column
        if 'quality' in columns:
            columns.remove('quality')
        if 'quality_label' in columns:
            columns.remove('quality_label')
    
    initial_shape = df.shape[0]
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Clip outliers instead of removing
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers > 0:
            logger.info(f"  {col}: {outliers} outlier(s) di-clip [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    logger.info(f"Shape setelah handle outlier: {df.shape}")
    
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Konversi kolom quality menjadi label biner.
    quality >= threshold -> 1 (good)
    quality < threshold -> 0 (bad)
    
    Args:
        df: DataFrame input
        threshold: Threshold untuk klasifikasi
        
    Returns:
        DataFrame dengan kolom quality_label
    """
    logger.info(f"Membuat target biner (threshold >= {threshold})...")
    
    df['quality_label'] = (df['quality'] >= threshold).astype(int)
    
    good_count = df['quality_label'].sum()
    bad_count = len(df) - good_count
    logger.info(f"  Good (1): {good_count}, Bad (0): {bad_count}")
    
    # Drop kolom quality original
    df = df.drop('quality', axis=1)
    
    return df


def feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Melakukan feature scaling menggunakan StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("Melakukan feature scaling (StandardScaler)...")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info("Feature scaling selesai.")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(df: pd.DataFrame, target_col: str = 'quality_label', 
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Membagi data menjadi training dan testing set.
    
    Args:
        df: DataFrame input
        target_col: Nama kolom target
        test_size: Proporsi test set
        random_state: Random seed
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Membagi data (test_size={test_size}, random_state={random_state})...")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"  Train target dist: {dict(y_train.value_counts())}")
    logger.info(f"  Test target dist: {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(input_path: str, output_dir: str) -> dict:
    """
    Menjalankan seluruh pipeline preprocessing.
    
    Args:
        input_path: Path ke raw dataset
        output_dir: Direktori output untuk menyimpan hasil
        
    Returns:
        Dictionary berisi metadata preprocessing
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 60)
    
    # 1. Load data
    df = load_data(input_path)
    
    # 2. EDA
    eda_results = explore_data(df)
    
    # 3. Handle missing values
    df = handle_missing_values(df)
    
    # 4. Remove duplicates
    df = remove_duplicates(df)
    
    # 5. Handle outliers
    df = handle_outliers(df)
    
    # 6. Create binary target
    df = create_binary_target(df, threshold=7)
    
    # 7. Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 8. Feature scaling
    X_train_scaled, X_test_scaled, scaler = feature_scaling(X_train, X_test)
    
    # 9. Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save scaler info
    scaler_info = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": X_train.columns.tolist()
    }
    with open(os.path.join(output_dir, 'scaler_info.json'), 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    # Save metadata
    metadata = {
        "input_path": input_path,
        "output_dir": output_dir,
        "original_shape": list(eda_results["shape"]),
        "final_shape": list(df.shape),
        "features": X_train.columns.tolist(),
        "target": "quality_label",
        "train_size": X_train_scaled.shape[0],
        "test_size": X_test_scaled.shape[0],
        "preprocessing_steps": [
            "load_data",
            "handle_missing_values",
            "remove_duplicates",
            "handle_outliers (IQR clipping)",
            "create_binary_target (threshold=7)",
            "train_test_split (80/20, stratified)",
            "feature_scaling (StandardScaler)"
        ]
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("PIPELINE PREPROCESSING SELESAI")
    logger.info(f"Output disimpan di: {output_dir}")
    logger.info("=" * 60)
    
    return metadata


if __name__ == "__main__":
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, '..', 'wine_quality_raw', 'winequality-red.csv')
    output_dir = os.path.join(script_dir, 'wine_quality_preprocessing')
    
    # Run pipeline
    metadata = run_preprocessing_pipeline(input_path, output_dir)
    
    print("\n[OK] Preprocessing selesai!")
    print(f"   Train samples: {metadata['train_size']}")
    print(f"   Test samples: {metadata['test_size']}")
    print(f"   Features: {len(metadata['features'])}")
