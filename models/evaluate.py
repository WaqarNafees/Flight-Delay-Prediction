"""
Evaluation Module
Comprehensive model evaluation and reporting
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, classification_report
)

logger = logging.getLogger(__name__)
REPORTS_PATH = "reports/"


def plot_roc_curve(model, X_test, y_test, model_name: str = "Model",
                   save_path: str = None):
    """Plot ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2563EB', lw=2,
            label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2563EB')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(model, X_test, y_test,
                           model_name: str = "Model", save_path: str = None):
    """Plot confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['On Time', 'Delayed'],
                yticklabels=['On Time', 'Delayed'], ax=ax)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(fi_df: pd.DataFrame,
                             top_n: int = 10, save_path: str = None):
    """Plot top N feature importances."""
    if fi_df.empty:
        return None
    
    top = fi_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2563EB' if i == 0 else '#93C5FD' for i in range(len(top))]
    bars = ax.barh(top['feature'][::-1], top['importance'][::-1],
                    color=colors[::-1])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(all_results: dict, save_path: str = None):
    """Bar chart comparing all models across metrics."""
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    
    model_names = list(all_results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706']
    
    for i, (name, results) in enumerate(all_results.items()):
        vals = [results[m] for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                       label=name, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_full_report(results: dict, output_dir: str = REPORTS_PATH):
    """Generate and save all evaluation plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    best_model = results['best_model']
    best_name = results['best_name']
    X_test = results['X_test']
    y_test = results['y_test']
    
    logger.info("Generating evaluation reports...")
    
    plot_roc_curve(best_model, X_test, y_test, best_name,
                    save_path=f"{output_dir}/roc_curve.png")
    
    plot_confusion_matrix(best_model, X_test, y_test, best_name,
                           save_path=f"{output_dir}/confusion_matrix.png")
    
    plot_model_comparison(results['all_results'],
                           save_path=f"{output_dir}/model_comparison.png")
    
    # Text report
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred,
                                    target_names=['On Time', 'Delayed'])
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write(f"Best Model: {best_name}\n\n")
        f.write(report)
        f.write(f"\n\nAll Results:\n")
        for name, m in results['all_results'].items():
            f.write(f"\n{name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
    
    logger.info(f"Reports saved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from etl.ingest import load_or_generate
    from etl.clean import clean
    from features.engineer import build_full_feature_pipeline
    from models.train import train_all_models, save_model
    
    raw = load_or_generate(n_samples=20000)
    cleaned = clean(raw)
    X, y, _ = build_full_feature_pipeline(cleaned)
    results = train_all_models(X, y)
    save_model(results['best_model'], metrics=results['best_metrics'],
               model_name=results['best_name'])
    generate_full_report(results)
    print("✅ Evaluation complete")
