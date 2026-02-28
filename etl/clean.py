"""
ETL Module: Data Cleaning
Handles all data cleaning and validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, delay_threshold: int = 15) -> pd.DataFrame:
    """
    Full cleaning pipeline for BTS flight data.
    
    Args:
        df: Raw flight DataFrame
        delay_threshold: Minutes threshold to define a delay (default: 15)
    
    Returns:
        Cleaned DataFrame with target label
    """
    logger.info(f"Starting cleaning on {len(df):,} rows")
    original_len = len(df)
    
    # 1. Strip column whitespace
    df.columns = df.columns.str.strip()
    
    # 2. Parse date column
    if 'FL_DATE' in df.columns:
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    
    # 3. Convert numeric columns
    numeric_cols = ['DEP_DELAY', 'ARR_DELAY', 'DISTANCE', 'DEP_TIME',
                    'WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY',
                    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Drop rows with missing target
    df = df.dropna(subset=['ARR_DELAY'])
    logger.info(f"After dropping null ARR_DELAY: {len(df):,} rows")
    
    # 5. Remove cancelled / diverted flights (extremely late outliers)
    df = df[df['ARR_DELAY'] < 600]
    df = df[df['ARR_DELAY'] > -120]
    
    # 6. Fill delay-type columns with 0 (NA = no delay of that type)
    delay_type_cols = ['WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY',
                        'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    for col in delay_type_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 7. Fill DEP_DELAY with 0 if missing (assumed on-time)
    if 'DEP_DELAY' in df.columns:
        df['DEP_DELAY'] = df['DEP_DELAY'].fillna(0)
    
    # 8. Create binary target label
    df['is_delayed'] = (df['ARR_DELAY'] > delay_threshold).astype(int)
    
    # 9. Extract temporal features from FL_DATE
    if 'FL_DATE' in df.columns and pd.api.types.is_datetime64_any_dtype(df['FL_DATE']):
        df['MONTH'] = df['FL_DATE'].dt.month
        if 'DAY_OF_WEEK' not in df.columns:
            df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1
    else:
        if 'MONTH' not in df.columns:
            df['MONTH'] = 1
    
    # 10. Extract hour from DEP_TIME (format: HHMM)
    if 'DEP_TIME' in df.columns:
        df['HOUR'] = (df['DEP_TIME'] // 100).clip(0, 23).fillna(12).astype(int)
    else:
        df['HOUR'] = 12
    
    # 11. Reset index
    df = df.reset_index(drop=True)
    
    removed = original_len - len(df)
    delay_rate = df['is_delayed'].mean()
    
    logger.info(f"Cleaning complete: {len(df):,} rows retained, "
                f"{removed:,} removed ({removed/original_len:.1%})")
    logger.info(f"Delay rate (>{delay_threshold}min): {delay_rate:.1%}")
    
    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate cleaned DataFrame has required columns and is non-empty."""
    required = ['ARR_DELAY', 'is_delayed', 'DAY_OF_WEEK', 'MONTH']
    
    for col in required:
        if col not in df.columns:
            return False, f"Missing required column: {col}"
    
    if len(df) == 0:
        return False, "DataFrame is empty after cleaning"
    
    if df['is_delayed'].nunique() < 2:
        return False, "Target column has only one class — check data"
    
    return True, "Validation passed"


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for logging/reporting."""
    return {
        "total_flights": len(df),
        "delayed_flights": int(df['is_delayed'].sum()),
        "delay_rate": float(df['is_delayed'].mean()),
        "avg_arr_delay": float(df['ARR_DELAY'].mean()),
        "airlines": df['OP_CARRIER'].nunique() if 'OP_CARRIER' in df.columns else 0,
        "routes": (df['ORIGIN'] + '-' + df['DEST']).nunique()
            if all(c in df.columns for c in ['ORIGIN', 'DEST']) else 0
    }


if __name__ == "__main__":
    from etl.ingest import load_or_generate
    raw = load_or_generate()
    cleaned = clean(raw)
    valid, msg = validate_dataframe(cleaned)
    print(f"Validation: {msg}")
    print(f"\nSummary: {get_data_summary(cleaned)}")
    print(f"\nCleaned sample:\n{cleaned.head()}")
