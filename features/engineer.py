"""
Feature Engineering Module
Builds features for the flight delay prediction model
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

# Features used in training — must match at inference time
NUMERIC_FEATURES = ['DEP_DELAY', 'DISTANCE', 'HOUR']
CATEGORICAL_FEATURES = ['DAY_OF_WEEK', 'MONTH', 'DIST_BUCKET', 'IS_WEEKEND',
                         'IS_PEAK_HOUR', 'SEASON']
LABEL_ENCODED_FEATURES = ['AIRLINE_ENC', 'ORIGIN_ENC', 'DEST_ENC']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + LABEL_ENCODED_FEATURES

# Encoder storage paths
ENCODER_PATH = "models/encoders.pkl"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from cleaned DataFrame.
    Includes temporal, geographic, and interaction features.
    """
    df = df.copy()
    
    # --- Distance buckets ---
    df['DIST_BUCKET'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 250, 500, 1000, 1500, 2000, 6000],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float).fillna(2)
    
    # --- Weekend flag ---
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    
    # --- Peak hour (morning rush: 6-9, evening rush: 16-20) ---
    df['IS_PEAK_HOUR'] = df['HOUR'].apply(
        lambda h: 1 if (6 <= h <= 9 or 16 <= h <= 20) else 0
    )
    
    # --- Season ---
    month_to_season = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                        6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['SEASON'] = df['MONTH'].map(month_to_season).fillna(0).astype(int)
    
    # --- Fill missing HOUR ---
    if 'HOUR' not in df.columns:
        df['HOUR'] = 12
    df['HOUR'] = df['HOUR'].fillna(12).astype(int)
    
    # --- DEP_DELAY bins (no data leakage — dep delay known at gate) ---
    df['DEP_DELAY'] = df['DEP_DELAY'].fillna(0)
    
    logger.info(f"Feature engineering complete: {len(df):,} rows, "
                f"{len(ALL_FEATURES)} features")
    return df


def encode_categoricals(df: pd.DataFrame,
                          fit: bool = True,
                          encoder_path: str = ENCODER_PATH) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode airline, origin, destination.
    
    Args:
        df: DataFrame with OP_CARRIER, ORIGIN, DEST columns
        fit: If True, fit new encoders and save. If False, load existing.
        encoder_path: Path to save/load encoders
    
    Returns:
        DataFrame with encoded columns, encoders dict
    """
    if fit:
        encoders = {}
        
        for col, enc_col in [('OP_CARRIER', 'AIRLINE_ENC'),
                               ('ORIGIN', 'ORIGIN_ENC'),
                               ('DEST', 'DEST_ENC')]:
            if col in df.columns:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                logger.info(f"Encoded {col}: {le.classes_[:5]}... "
                            f"({len(le.classes_)} classes)")
            else:
                df[enc_col] = 0
        
        # Save encoders
        Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, encoder_path)
        logger.info(f"Encoders saved to {encoder_path}")
        
    else:
        encoders = joblib.load(encoder_path)
        
        for col, enc_col in [('OP_CARRIER', 'AIRLINE_ENC'),
                               ('ORIGIN', 'ORIGIN_ENC'),
                               ('DEST', 'DEST_ENC')]:
            if col in df.columns and col in encoders:
                le = encoders[col]
                # Handle unseen categories
                df[enc_col] = df[col].astype(str).map(
                    dict(zip(le.classes_, le.transform(le.classes_)))
                ).fillna(0).astype(int)
            else:
                df[enc_col] = 0
    
    return df, encoders


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from engineered DataFrame."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    
    if missing:
        logger.warning(f"Missing features (will use 0): {missing}")
        for f in missing:
            df[f] = 0
    
    X = df[ALL_FEATURES].copy()
    y = df['is_delayed'].copy()
    
    logger.info(f"Feature matrix: X={X.shape}, y={y.shape}, "
                f"positive_rate={y.mean():.1%}")
    return X, y


def build_full_feature_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Run full feature pipeline: engineer + encode + extract."""
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, fit=True)
    X, y = get_feature_matrix(df)
    return X, y, encoders


def encode_single_record(record: dict,
                          encoders: dict,
                          encoder_path: str = ENCODER_PATH) -> pd.DataFrame:
    """
    Encode a single inference record.
    Used at prediction time via the API.
    
    Args:
        record: Dict with raw feature values
        encoders: Loaded encoder dict
        encoder_path: Path to encoders if not passed
    
    Returns:
        DataFrame row ready for model prediction
    """
    if encoders is None:
        encoders = joblib.load(encoder_path)
    
    df = pd.DataFrame([record])
    
    # Apply same feature engineering
    df = engineer_features(df)
    
    # Encode categoricals
    for col, enc_col in [('OP_CARRIER', 'AIRLINE_ENC'),
                           ('ORIGIN', 'ORIGIN_ENC'),
                           ('DEST', 'DEST_ENC')]:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = df[col].astype(str).iloc[0]
            if val in le.classes_:
                df[enc_col] = le.transform([val])[0]
            else:
                df[enc_col] = 0
        else:
            df[enc_col] = 0
    
    # Ensure all features present
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = 0
    
    return df[ALL_FEATURES]


def get_airline_list(encoder_path: str = ENCODER_PATH) -> List[str]:
    """Return list of known airlines from encoders."""
    try:
        encoders = joblib.load(encoder_path)
        return sorted(encoders['OP_CARRIER'].classes_.tolist())
    except Exception:
        return ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9']


def get_airport_list(encoder_path: str = ENCODER_PATH) -> List[str]:
    """Return list of known airports from encoders."""
    try:
        encoders = joblib.load(encoder_path)
        return sorted(encoders['ORIGIN'].classes_.tolist())
    except Exception:
        return ['ATL', 'ORD', 'DFW', 'LAX', 'JFK', 'SFO', 'SEA', 'DEN']


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from etl.ingest import load_or_generate
    from etl.clean import clean
    
    raw = load_or_generate()
    cleaned = clean(raw)
    X, y, encoders = build_full_feature_pipeline(cleaned)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"\nSample:\n{X.head()}")
