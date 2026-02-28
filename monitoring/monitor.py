"""
Monitoring Module
Tracks prediction distribution and data drift
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PATH = "data/inference_logs.csv"
REPORTS_PATH = "reports/"


def load_inference_logs(path: str = LOG_PATH) -> Optional[pd.DataFrame]:
    """Load inference log CSV."""
    if not os.path.exists(path):
        logger.warning(f"No inference logs found at {path}")
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def compute_prediction_stats(df: pd.DataFrame) -> dict:
    """Compute stats on inference log."""
    if df is None or len(df) == 0:
        return {}
    
    return {
        "total_predictions": len(df),
        "delay_rate": float((df['prediction'] == 'Delayed').mean()),
        "avg_probability": float(df['probability'].mean()),
        "high_risk_count": int((df['probability'] > 0.7).sum()),
        "date_range": {
            "start": str(df['timestamp'].min()),
            "end": str(df['timestamp'].max())
        }
    }


def plot_prediction_distribution(df: pd.DataFrame,
                                   save_path: str = None):
    """Plot distribution of delay probabilities."""
    if df is None or len(df) == 0:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Probability histogram
    axes[0].hist(df['probability'], bins=30, color='#2563EB',
                  alpha=0.8, edgecolor='white')
    axes[0].axvline(0.5, color='red', linestyle='--', lw=2, label='Decision boundary')
    axes[0].set_xlabel('Delay Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Prediction Probability Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prediction pie chart
    counts = df['prediction'].value_counts()
    colors = ['#16A34A', '#DC2626']
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Prediction Split', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def check_prediction_drift(df: pd.DataFrame,
                             baseline_delay_rate: float = 0.22,
                             window_days: int = 7,
                             threshold: float = 0.1) -> dict:
    """
    Check if recent predictions have drifted from training baseline.
    
    Args:
        df: Inference log DataFrame
        baseline_delay_rate: Expected delay rate from training data
        window_days: Number of recent days to check
        threshold: Alert if drift exceeds this value
    """
    if df is None or len(df) == 0:
        return {"drift_detected": False, "message": "No data"}
    
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    recent = df[df['timestamp'] >= cutoff]
    
    if len(recent) < 50:
        return {"drift_detected": False,
                "message": f"Insufficient recent data ({len(recent)} predictions)"}
    
    recent_delay_rate = (recent['prediction'] == 'Delayed').mean()
    drift = abs(recent_delay_rate - baseline_delay_rate)
    
    return {
        "drift_detected": drift > threshold,
        "baseline_delay_rate": baseline_delay_rate,
        "recent_delay_rate": float(recent_delay_rate),
        "drift_magnitude": float(drift),
        "threshold": threshold,
        "window_days": window_days,
        "recent_predictions": len(recent),
        "alert": f"⚠️ Data drift detected! ({drift:.1%} > {threshold:.1%})"
                  if drift > threshold else "✅ No significant drift",
        "recommendation": "Consider retraining model" if drift > threshold else "Model healthy"
    }


def generate_monitoring_report(output_dir: str = REPORTS_PATH) -> str:
    """Generate full monitoring report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = load_inference_logs()
    
    if df is None:
        logger.info("No inference logs to report on")
        return output_dir
    
    stats = compute_prediction_stats(df)
    drift = check_prediction_drift(df)
    
    plot_prediction_distribution(
        df,
        save_path=f"{output_dir}/prediction_distribution.png"
    )
    
    # Write monitoring summary
    report = f"""
MONITORING REPORT
Generated: {datetime.utcnow().isoformat()}
{'='*50}

PREDICTION STATS:
  Total Predictions: {stats.get('total_predictions', 0)}
  Delay Rate:        {stats.get('delay_rate', 0):.1%}
  Avg Probability:   {stats.get('avg_probability', 0):.3f}
  High Risk (>70%):  {stats.get('high_risk_count', 0)}

DRIFT ANALYSIS:
  Status:            {drift.get('alert', 'N/A')}
  Baseline Rate:     {drift.get('baseline_delay_rate', 0):.1%}
  Recent Rate:       {drift.get('recent_delay_rate', 0):.1%}
  Drift:             {drift.get('drift_magnitude', 0):.1%}
  Recommendation:    {drift.get('recommendation', 'N/A')}
"""
    
    with open(f"{output_dir}/monitoring_report.txt", 'w') as f:
        f.write(report)
    
    logger.info(f"Monitoring report saved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    generate_monitoring_report()
    print("✅ Monitoring report generated")
