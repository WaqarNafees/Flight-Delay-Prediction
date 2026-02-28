"""
Model Training Module
Trains multiple models, tracks with MLflow, saves best model
"""

import os
import sys
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logging.warning("MLflow not installed - tracking disabled. pip install mlflow")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM not available")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/pipeline.pkl"
METRICS_PATH = "models/metrics.json"
MLFLOW_TRACKING_URI = "mlflow_tracking"


def get_models() -> Dict[str, Any]:
    """Return dict of models to compare."""
    models = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000, random_state=42, C=0.1
            ))
        ]),
        "RandomForest": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5, n_jobs=-1, random_state=42
            ))
        ]),
    }
    
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False,
                eval_metric='logloss', random_state=42, n_jobs=-1
            ))
        ])
    
    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbose=-1
            ))
        ])
    
    return models


def evaluate_model(model, X_test: pd.DataFrame,
                    y_test: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def train_all_models(X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Dict:
    """
    Train all models, track with MLflow, return best.
    
    Returns:
        dict with keys: best_model, best_name, best_metrics,
                        all_results, X_test, y_test
    """
    logger.info(f"Starting training on {len(X):,} samples, "
                f"{X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # MLflow setup
    if HAS_MLFLOW:
        Path(MLFLOW_TRACKING_URI).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_TRACKING_URI)}")
        mlflow.set_experiment("flight_delay_prediction")
    
    models = get_models()
    all_results = {}
    best_model = None
    best_name = None
    best_auc = 0
    
    for name, pipeline in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        
        if HAS_MLFLOW:
            ctx = mlflow.start_run(run_name=name)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()
        
        with ctx:
            # Train
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(pipeline, X_test, y_test)
            all_results[name] = metrics
            
            # Log to MLflow
            if HAS_MLFLOW:
                mlflow.log_params({
                    "model_type": name,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "n_features": X.shape[1],
                    "positive_rate": float(y.mean())
                })
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    pipeline, "model",
                    registered_model_name=f"FlightDelay_{name}"
                )
            
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
                        f"F1: {metrics['f1']:.4f} | "
                        f"Recall: {metrics['recall']:.4f}")
            
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = pipeline
                best_name = name
                best_metrics = metrics
    
    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_name} | ROC-AUC: {best_auc:.4f}")
    
    return {
        "best_model": best_model,
        "best_name": best_name,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train
    }


def save_model(model, path: str = MODEL_PATH,
               metrics: dict = None, model_name: str = ""):
    """Save model pipeline and metrics."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    
    if metrics:
        meta = {
            "model_name": model_name,
            "metrics": metrics,
            "feature_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
        }
        with open(METRICS_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metrics saved to {METRICS_PATH}")


def load_model(path: str = MODEL_PATH):
    """Load saved model pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Run training first.")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    try:
        # Get the actual model from pipeline
        clf = model.named_steps.get('model', model)
        
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        else:
            return pd.DataFrame()
        
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return fi_df
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    sys.path.insert(0, '.')
    from etl.ingest import load_or_generate
    from etl.clean import clean
    from features.engineer import build_full_feature_pipeline
    
    logger.info("Starting full training pipeline...")
    
    raw = load_or_generate(n_samples=30000)
    cleaned = clean(raw)
    X, y, encoders = build_full_feature_pipeline(cleaned)
    
    results = train_all_models(X, y)
    
    save_model(
        results['best_model'],
        metrics=results['best_metrics'],
        model_name=results['best_name']
    )
    
    print("\n📊 All Model Results:")
    for name, metrics in results['all_results'].items():
        print(f"  {name:20s} | AUC: {metrics['roc_auc']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
    
    print(f"\n✅ Best: {results['best_name']} saved to {MODEL_PATH}")
