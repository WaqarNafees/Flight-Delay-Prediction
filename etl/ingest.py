"""
ETL Module: Data Ingestion
Handles downloading and loading BTS flight data
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BTS column mapping
REQUIRED_COLUMNS = [
    'FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
    'DEP_TIME', 'DEP_DELAY', 'ARR_DELAY',
    'DISTANCE', 'DAY_OF_WEEK',
    'WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

AIRLINE_MAP = {
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air',
    'HA': 'Hawaiian Airlines',
    '9E': 'Endeavor Air',
    'MQ': 'Envoy Air',
    'OH': 'PSA Airlines',
    'OO': 'SkyWest Airlines',
    'YX': 'Republic Airways',
    'YV': 'Mesa Airlines'
}


def load_csv(filepath: str) -> pd.DataFrame:
    """Load BTS CSV file into DataFrame."""
    logger.info(f"Loading data from: {filepath}")
    
    df = pd.read_csv(
        filepath,
        usecols=lambda c: c in REQUIRED_COLUMNS or c.strip() in REQUIRED_COLUMNS,
        low_memory=False
    )
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def generate_synthetic_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic flight data for demo/testing.
    Used when actual BTS data is not available (e.g., HuggingFace demo).
    Based on real statistical distributions from BTS dataset.
    """
    logger.info(f"Generating {n_samples:,} synthetic flight records...")
    np.random.seed(seed)

    airlines = list(AIRLINE_MAP.keys())
    origins = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'CLT', 'LAS', 'PHX',
               'MCO', 'SEA', 'MSP', 'DTW', 'BOS', 'EWR', 'JFK', 'SFO',
               'LGA', 'FLL', 'IAD', 'IAH', 'MIA', 'MDW', 'PHL', 'BWI', 'SLC']
    
    n = n_samples
    
    # Realistic departure delay distribution (mixture model)
    dep_delay = np.where(
        np.random.random(n) < 0.75,
        np.random.normal(-2, 8, n),        # On-time flights (mostly early/on-time)
        np.random.exponential(25, n) + 15  # Delayed flights
    ).clip(-30, 300).round(1)
    
    # Arrival delay correlated with departure delay
    arr_delay = (dep_delay * 0.85 + np.random.normal(0, 10, n)).clip(-60, 360).round(1)
    
    # Distance based on real route distributions
    distances = np.random.choice(
        [150, 300, 500, 750, 1000, 1200, 1500, 2000, 2500],
        n,
        p=[0.08, 0.15, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05, 0.02]
    ) + np.random.normal(0, 50, n)
    distances = distances.clip(50, 5000).round(0)
    
    # Departure time (hour of day) — 19 hours (5..23)
    hour_probs = np.array([0.04, 0.06, 0.08, 0.09, 0.09, 0.08, 0.07, 0.06,
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.03, 0.02])
    hour_probs = hour_probs / hour_probs.sum()  # Ensure sum == 1
    dep_hours = np.random.choice(range(5, 24), n, p=hour_probs)
    
    df = pd.DataFrame({
        'FL_DATE': pd.date_range('2023-01-01', periods=n, freq='1min').date,
        'OP_CARRIER': np.random.choice(airlines, n, p=[0.15, 0.14, 0.12, 0.13,
                                                        0.06, 0.05, 0.05, 0.04,
                                                        0.03, 0.03, 0.04, 0.04,
                                                        0.04, 0.04, 0.02, 0.02]),
        'ORIGIN': np.random.choice(origins, n),
        'DEST': np.random.choice(origins, n),
        'DEP_TIME': dep_hours * 100 + np.random.randint(0, 60, n),
        'DEP_DELAY': dep_delay,
        'ARR_DELAY': arr_delay,
        'DISTANCE': distances,
        'DAY_OF_WEEK': np.random.randint(1, 8, n),
        'WEATHER_DELAY': np.where(np.random.random(n) < 0.08,
                                   np.random.exponential(15, n), 0).round(1),
        'CARRIER_DELAY': np.where(np.random.random(n) < 0.12,
                                   np.random.exponential(20, n), 0).round(1),
        'NAS_DELAY': np.where(np.random.random(n) < 0.10,
                               np.random.exponential(12, n), 0).round(1),
        'SECURITY_DELAY': np.where(np.random.random(n) < 0.01,
                                    np.random.exponential(5, n), 0).round(1),
        'LATE_AIRCRAFT_DELAY': np.where(np.random.random(n) < 0.15,
                                         np.random.exponential(18, n), 0).round(1),
    })
    
    # Remove same origin-dest
    same_route = df['ORIGIN'] == df['DEST']
    dest_shifted = np.roll(df['DEST'].values, 1)
    df.loc[same_route, 'DEST'] = dest_shifted[same_route]
    
    logger.info(f"Generated data - Delay rate: {(df['ARR_DELAY'] > 15).mean():.1%}")
    return df


def save_to_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} rows to {path}")


def load_or_generate(raw_path: str = "data/raw/flights.csv",
                     n_samples: int = 20000) -> pd.DataFrame:
    """
    Load existing data or generate synthetic if not found.
    The bundled data/raw/flights.csv ships with the project —
    mirrors BTS On-Time Performance Dataset distributions.
    """
    # Check bundled path first, then fallback paths
    search_paths = [
        raw_path,
        "data/raw/flights.csv",
        "/app/data/raw/flights.csv",          # HuggingFace container path
        "/home/user/app/data/raw/flights.csv"  # HuggingFace symlink path
    ]
    for path in search_paths:
        if os.path.exists(path):
            logger.info(f"Loading bundled dataset from {path}")
            return load_csv(path)

    logger.info("No bundled data found — generating synthetic dataset")
    df = generate_synthetic_data(n_samples)
    save_to_csv(df, raw_path)
    return df


if __name__ == "__main__":
    df = load_or_generate()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDelay rate: {(df['ARR_DELAY'] > 15).mean():.1%}")
