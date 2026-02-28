"""
✈️ Flight Delay Prediction — Hugging Face Spaces App
Full ML pipeline: Train → Evaluate → Predict
"""

import os
import sys
import json
import warnings
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Lazy imports (so Gradio loads quickly) ─────────────────────────────────

_model = None
_encoders = None
_model_info = {}
_training_results = None

def get_model():
    global _model
    if _model is None and os.path.exists("models/pipeline.pkl"):
        import joblib
        _model = joblib.load("models/pipeline.pkl")
    return _model

def get_encoders():
    global _encoders
    if _encoders is None and os.path.exists("models/encoders.pkl"):
        import joblib
        _encoders = joblib.load("models/encoders.pkl")
    return _encoders

def get_model_info():
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            return json.load(f)
    return {}


# ── Airlines & Airports ────────────────────────────────────────────────────

AIRLINES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "9E": "Endeavor Air",
    "OO": "SkyWest Airlines",
    "MQ": "Envoy Air",
    "OH": "PSA Airlines",
    "YX": "Republic Airways",
    "YV": "Mesa Airlines"
}

AIRPORTS = [
    "ATL", "ORD", "DFW", "DEN", "LAX", "CLT", "LAS", "PHX",
    "MCO", "SEA", "MSP", "DTW", "BOS", "EWR", "JFK", "SFO",
    "LGA", "FLL", "IAD", "IAH", "MIA", "MDW", "PHL", "BWI", "SLC",
    "PDX", "HOU", "OAK", "AUS", "SMF", "RSW", "TPA", "SAN", "MSY"
]

AIRLINE_CHOICES = [f"{code} — {name}" for code, name in AIRLINES.items()]
AIRPORT_CHOICES = sorted(AIRPORTS)

DAYS = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday",
    4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"
}
MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


# ── Training Pipeline ──────────────────────────────────────────────────────

def run_training_pipeline(n_samples: int, progress=gr.Progress(track_tqdm=True)):
    """Full pipeline: generate data → clean → features → train → evaluate."""
    global _model, _encoders, _model_info, _training_results
    
    logs = []
    
    def log(msg):
        logs.append(msg)
        logger.info(msg)
        return "\n".join(logs)
    
    try:
        # Step 1: Load or Generate Data
        progress(0.05, desc="Loading flight data...")

        from etl.ingest import load_or_generate
        raw = load_or_generate(n_samples=int(n_samples))
        log(f"✅ Data loaded: {len(raw):,} flights")
        log(f"   Source: {'Bundled CSV (data/raw/flights.csv)' if os.path.exists('data/raw/flights.csv') else 'Synthetic generator'}")
        
        # Step 2: Clean
        progress(0.2, desc="Cleaning data...")
        log("\n🧹 Running ETL cleaning pipeline...")
        
        from etl.clean import clean, get_data_summary
        cleaned = clean(raw)
        summary = get_data_summary(cleaned)
        log(f"✅ Cleaned: {summary['total_flights']:,} flights retained")
        log(f"   Delay rate (>15min): {summary['delay_rate']:.1%}")
        log(f"   Airlines: {summary['airlines']} | Routes: {summary['routes']}")
        
        # Step 3: Feature Engineering
        progress(0.35, desc="Engineering features...")
        log("\n⚙️  Building feature matrix...")
        
        from features.engineer import build_full_feature_pipeline
        X, y, encoders = build_full_feature_pipeline(cleaned)
        log(f"✅ Features: {X.shape[1]} features × {len(X):,} samples")
        log(f"   Features: {', '.join(X.columns.tolist())}")
        
        # Step 4: Train Models
        progress(0.5, desc="Training models (this takes ~30-60s)...")
        log("\n🤖 Training models with MLflow tracking...")
        
        from models.train import train_all_models, save_model
        results = train_all_models(X, y)
        _training_results = results
        
        log(f"\n📊 Model Comparison:")
        for name, metrics in results['all_results'].items():
            log(f"   {name:20s} | AUC: {metrics['roc_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")
        
        log(f"\n🏆 Best Model: {results['best_name']} "
            f"(ROC-AUC: {results['best_metrics']['roc_auc']:.4f})")
        
        # Step 5: Save
        progress(0.8, desc="Saving model...")
        save_model(results['best_model'],
                   metrics=results['best_metrics'],
                   model_name=results['best_name'])
        
        # Reload
        import joblib
        _model = joblib.load("models/pipeline.pkl")
        _encoders = joblib.load("models/encoders.pkl")
        _model_info = get_model_info()
        
        log(f"✅ Model saved to models/pipeline.pkl")
        
        # Step 6: Generate plots
        progress(0.9, desc="Generating evaluation plots...")
        log("\n📈 Generating evaluation report...")
        
        from models.evaluate import (plot_roc_curve, plot_confusion_matrix,
                                      plot_model_comparison, plot_feature_importance)
        from features.engineer import ALL_FEATURES, get_feature_importance
        from models.train import get_feature_importance as get_fi
        
        roc_fig = plot_roc_curve(results['best_model'], results['X_test'],
                                   results['y_test'], results['best_name'])
        cm_fig = plot_confusion_matrix(results['best_model'], results['X_test'],
                                        results['y_test'], results['best_name'])
        comp_fig = plot_model_comparison(results['all_results'])
        
        fi_df = get_fi(results['best_model'], ALL_FEATURES)
        fi_fig = plot_feature_importance(fi_df) if not fi_df.empty else None
        
        progress(1.0, desc="Done!")
        log(f"\n✅ Training pipeline complete!")
        log(f"   → Switch to 'Make Prediction' tab to predict delays")
        
        return ("\n".join(logs), roc_fig, cm_fig, comp_fig,
                fi_fig if fi_fig else plt.figure())
    
    except Exception as e:
        log(f"\n❌ Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return ("\n".join(logs), None, None, None, None)


# ── Prediction ─────────────────────────────────────────────────────────────

def predict_delay(airline_str, origin, destination, dep_delay,
                   distance, day_str, month_str, hour):
    """Single flight prediction."""
    model = get_model()
    encoders = get_encoders()
    
    if model is None:
        return (
            "⚠️ **Model not trained yet!**\n\nGo to the **Train Model** tab "
            "and click **Start Training Pipeline** first.",
            None
        )
    
    try:
        # Parse inputs
        airline_code = airline_str.split(" — ")[0].strip()
        day_num = [k for k, v in DAYS.items() if v in day_str][0]
        month_num = [k for k, v in MONTHS.items() if v in month_str][0]
        
        record = {
            "OP_CARRIER": airline_code,
            "ORIGIN": origin,
            "DEST": destination,
            "DEP_DELAY": float(dep_delay),
            "DISTANCE": float(distance),
            "DAY_OF_WEEK": int(day_num),
            "MONTH": int(month_num),
            "HOUR": int(hour),
            "DEP_TIME": int(hour) * 100
        }
        
        from features.engineer import encode_single_record
        X = encode_single_record(record, encoders)
        
        prob = float(model.predict_proba(X)[0][1])
        prediction = "Delayed 🔴" if prob >= 0.5 else "On Time 🟢"
        
        # Risk level
        if prob < 0.25:
            risk = "🟢 LOW RISK"
            color_note = "This flight has a low probability of delay."
        elif prob < 0.45:
            risk = "🟡 MODERATE RISK"
            color_note = "Some chance of delay. Monitor flight status."
        elif prob < 0.65:
            risk = "🟠 HIGH RISK"
            color_note = "Significant chance of delay. Allow buffer time."
        else:
            risk = "🔴 VERY HIGH RISK"
            color_note = "Very likely to be delayed. Plan accordingly."
        
        info = get_model_info()
        model_name = info.get("model_name", "Unknown")
        
        result_text = f"""
## Prediction: {prediction}

| Field | Value |
|-------|-------|
| **Delay Probability** | {prob:.1%} |
| **Risk Level** | {risk} |
| **Route** | {origin} → {destination} |
| **Airline** | {AIRLINES.get(airline_code, airline_code)} |
| **Distance** | {distance:,.0f} miles |
| **Departure Delay** | {dep_delay:+.0f} min |
| **Model Used** | {model_name} |

> {color_note}
"""
        
        # Gauge plot
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Background segments
        segments = [(0, 0.25, '#16A34A', 'Low'),
                     (0.25, 0.45, '#CA8A04', 'Moderate'),
                     (0.45, 0.65, '#EA580C', 'High'),
                     (0.65, 1.0, '#DC2626', 'Very High')]
        
        for start, end, color, label in segments:
            ax.barh(0, end - start, left=start, height=0.5,
                    color=color, alpha=0.7)
        
        # Needle
        ax.axvline(prob, color='black', linewidth=3, ymin=0.1, ymax=0.9)
        ax.plot(prob, 0, 'v', color='black', markersize=12)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.8)
        ax.set_xlabel('Delay Probability', fontsize=12)
        ax.set_title(f'Risk Level: {prob:.1%} — {prediction}',
                      fontsize=13, fontweight='bold')
        
        # Labels
        for start, end, color, label in segments:
            mid = (start + end) / 2
            ax.text(mid, -0.35, label, ha='center', fontsize=9,
                    color='white', fontweight='bold')
        
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        
        return result_text, fig
    
    except Exception as e:
        import traceback
        return f"❌ Prediction error: {str(e)}\n\n{traceback.format_exc()}", None


def predict_batch_ui(csv_file):
    """Batch prediction from uploaded CSV."""
    model = get_model()
    encoders = get_encoders()
    
    if model is None:
        return "⚠️ Model not trained. Run training first.", None
    
    if csv_file is None:
        return "Please upload a CSV file.", None
    
    try:
        df = pd.read_csv(csv_file.name)
        required = ['airline', 'origin', 'destination',
                     'distance', 'day_of_week', 'month']
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            return f"❌ Missing columns: {missing}\nRequired: {required}", None
        
        results = []
        from features.engineer import encode_single_record
        
        for _, row in df.iterrows():
            record = {
                "OP_CARRIER": str(row.get('airline', 'AA')).upper(),
                "ORIGIN": str(row.get('origin', 'ATL')).upper(),
                "DEST": str(row.get('destination', 'ORD')).upper(),
                "DEP_DELAY": float(row.get('departure_delay', 0)),
                "DISTANCE": float(row.get('distance', 1000)),
                "DAY_OF_WEEK": int(row.get('day_of_week', 3)),
                "MONTH": int(row.get('month', 6)),
                "HOUR": int(row.get('hour', 12)),
                "DEP_TIME": int(row.get('hour', 12)) * 100
            }
            X = encode_single_record(record, encoders)
            prob = float(model.predict_proba(X)[0][1])
            results.append({
                "Route": f"{record['ORIGIN']}→{record['DEST']}",
                "Airline": record['OP_CARRIER'],
                "Delay Probability": f"{prob:.1%}",
                "Prediction": "🔴 Delayed" if prob >= 0.5 else "🟢 On Time",
                "Risk": "HIGH" if prob >= 0.5 else "LOW"
            })
        
        result_df = pd.DataFrame(results)
        summary = (f"✅ Processed {len(result_df)} flights | "
                    f"Delayed: {(result_df['Prediction'].str.contains('Delayed')).sum()} | "
                    f"On Time: {(result_df['Prediction'].str.contains('On Time')).sum()}")
        
        return summary, result_df
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def get_model_status():
    """Check if model exists and return status."""
    if os.path.exists("models/pipeline.pkl"):
        info = get_model_info()
        name = info.get("model_name", "Unknown")
        metrics = info.get("metrics", {})
        auc = metrics.get("roc_auc", "N/A")
        f1 = metrics.get("f1", "N/A")
        return f"✅ **Model Ready** — {name} | ROC-AUC: {auc} | F1: {f1}"
    else:
        return "⚠️ **No model found.** Click 'Start Training Pipeline' to train."


def create_sample_csv():
    """Create a sample batch CSV for download."""
    sample = pd.DataFrame([
        {"airline": "AA", "origin": "JFK", "destination": "LAX",
         "departure_delay": 20, "distance": 2475, "day_of_week": 5,
         "month": 7, "hour": 14},
        {"airline": "DL", "origin": "ATL", "destination": "ORD",
         "departure_delay": 0, "distance": 718, "day_of_week": 2,
         "month": 3, "hour": 9},
        {"airline": "WN", "origin": "MDW", "destination": "DEN",
         "departure_delay": -5, "distance": 920, "day_of_week": 6,
         "month": 12, "hour": 16},
        {"airline": "UA", "origin": "SFO", "destination": "EWR",
         "departure_delay": 45, "distance": 2565, "day_of_week": 1,
         "month": 8, "hour": 7},
    ])
    path = "data/sample_batch.csv"
    os.makedirs("data", exist_ok=True)
    sample.to_csv(path, index=False)
    return path


# ── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    )
    
    with gr.Blocks(theme=theme, title="✈️ Flight Delay Prediction") as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <h1 style="font-size:2.2rem; font-weight:700; 
                       background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ✈️ Flight Delay Prediction Platform
            </h1>
            <p style="color:#64748b; font-size:1rem; margin-top:8px;">
                End-to-End ML System · ETL → Feature Engineering → MLflow → XGBoost/LightGBM → API
            </p>
            <div style="display:flex; justify-content:center; gap:12px; margin-top:12px; flex-wrap:wrap;">
                <span style="background:#dbeafe; color:#1e40af; padding:4px 12px; 
                             border-radius:20px; font-size:0.8rem; font-weight:600;">
                    📦 scikit-learn
                </span>
                <span style="background:#dcfce7; color:#166534; padding:4px 12px; 
                             border-radius:20px; font-size:0.8rem; font-weight:600;">
                    🚀 XGBoost / LightGBM
                </span>
                <span style="background:#fef3c7; color:#92400e; padding:4px 12px; 
                             border-radius:20px; font-size:0.8rem; font-weight:600;">
                    📊 MLflow Tracking
                </span>
                <span style="background:#f3e8ff; color:#6b21a8; padding:4px 12px; 
                             border-radius:20px; font-size:0.8rem; font-weight:600;">
                    🔌 FastAPI Backend
                </span>
            </div>
        </div>
        """)
        
        # Model status bar
        with gr.Row():
            status_box = gr.Markdown(value=get_model_status)
        
        with gr.Tabs() as tabs:
            
            # ── TAB 1: TRAIN ──────────────────────────────────────────────
            with gr.TabItem("🏋️ Train Model", id="train"):
                gr.Markdown("## Training Pipeline\nGenerates synthetic BTS data, runs ETL, engineers features, trains 4 models, tracks with MLflow.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        n_samples_slider = gr.Slider(
                            minimum=5000, maximum=100000, value=30000, step=5000,
                            label="Training Samples",
                            info="More samples = better model, longer training"
                        )
                        gr.Markdown("""
                        **What happens when you click Train:**
                        1. 📂 Load bundled BTS-style flight data (20K records)
                        2. 🧹 ETL: clean, validate, create delay label  
                        3. ⚙️ Feature engineering (12 features)  
                        4. 🤖 Train: Logistic Regression, Random Forest, XGBoost, LightGBM  
                        5. 📊 MLflow experiment tracking  
                        6. 🏆 Auto-select best model by ROC-AUC  
                        7. 💾 Save full sklearn Pipeline  
                        """)
                        train_btn = gr.Button(
                            "🚀 Start Training Pipeline",
                            variant="primary", size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        train_logs = gr.Textbox(
                            label="Training Logs",
                            lines=20, max_lines=30,
                            interactive=False,
                            placeholder="Training logs will appear here..."
                        )
                
                gr.Markdown("### 📊 Evaluation Results")
                with gr.Row():
                    roc_plot = gr.Plot(label="ROC Curve")
                    cm_plot = gr.Plot(label="Confusion Matrix")
                
                with gr.Row():
                    comp_plot = gr.Plot(label="Model Comparison")
                    fi_plot = gr.Plot(label="Feature Importance")
                
                train_btn.click(
                    fn=run_training_pipeline,
                    inputs=[n_samples_slider],
                    outputs=[train_logs, roc_plot, cm_plot, comp_plot, fi_plot]
                )
            
            # ── TAB 2: PREDICT ────────────────────────────────────────────
            with gr.TabItem("🎯 Make Prediction", id="predict"):
                gr.Markdown("## Single Flight Prediction\nPredict if a specific flight will be delayed more than 15 minutes.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        airline_input = gr.Dropdown(
                            choices=AIRLINE_CHOICES,
                            value="AA — American Airlines",
                            label="✈️ Airline"
                        )
                        with gr.Row():
                            origin_input = gr.Dropdown(
                                choices=AIRPORT_CHOICES, value="JFK",
                                label="🛫 Origin Airport"
                            )
                            dest_input = gr.Dropdown(
                                choices=AIRPORT_CHOICES, value="LAX",
                                label="🛬 Destination Airport"
                            )
                        
                        distance_input = gr.Slider(
                            minimum=50, maximum=5000, value=2475, step=50,
                            label="📏 Distance (miles)"
                        )
                        dep_delay_input = gr.Slider(
                            minimum=-30, maximum=180, value=0, step=5,
                            label="⏱️ Departure Delay (minutes)",
                            info="Use 0 if predicting before departure"
                        )
                        
                        with gr.Row():
                            day_input = gr.Dropdown(
                                choices=list(DAYS.values()),
                                value="Wednesday",
                                label="📅 Day of Week"
                            )
                            month_input = gr.Dropdown(
                                choices=list(MONTHS.values()),
                                value="July",
                                label="🗓️ Month"
                            )
                        
                        hour_input = gr.Slider(
                            minimum=0, maximum=23, value=14, step=1,
                            label="🕐 Departure Hour (0-23)"
                        )
                        
                        predict_btn = gr.Button(
                            "🔮 Predict Delay", variant="primary", size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        result_text = gr.Markdown(
                            value="*Make a prediction to see results*"
                        )
                        gauge_plot = gr.Plot(label="Risk Gauge")
                
                predict_btn.click(
                    fn=predict_delay,
                    inputs=[airline_input, origin_input, dest_input,
                             dep_delay_input, distance_input,
                             day_input, month_input, hour_input],
                    outputs=[result_text, gauge_plot]
                )
                
                # Example flights
                gr.Markdown("### 🎲 Try These Examples")
                gr.Examples(
                    examples=[
                        ["AA — American Airlines", "JFK", "LAX", 45, 2475, "Friday", "July", 17],
                        ["DL — Delta Air Lines", "ATL", "ORD", 0, 718, "Monday", "March", 8],
                        ["WN — Southwest Airlines", "MDW", "DEN", -5, 920, "Saturday", "December", 14],
                        ["UA — United Airlines", "SFO", "EWR", 90, 2565, "Monday", "August", 7],
                        ["B6 — JetBlue Airways", "BOS", "MCO", 10, 1258, "Thursday", "June", 19],
                    ],
                    inputs=[airline_input, origin_input, dest_input, dep_delay_input,
                             distance_input, day_input, month_input, hour_input],
                    label="Click any row to load example"
                )
            
            # ── TAB 3: BATCH PREDICT ──────────────────────────────────────
            with gr.TabItem("📋 Batch Predict", id="batch"):
                gr.Markdown("## Batch Prediction\nUpload a CSV file to predict delays for multiple flights at once.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        **CSV Format Required:**
                        ```
                        airline,origin,destination,departure_delay,distance,day_of_week,month,hour
                        AA,JFK,LAX,20,2475,5,7,14
                        DL,ATL,ORD,0,718,2,3,9
                        ```
                        """)
                        
                        sample_btn = gr.Button("📥 Download Sample CSV", variant="secondary")
                        sample_file = gr.File(label="Sample CSV", visible=False)
                        
                        upload = gr.File(label="📂 Upload CSV", file_types=[".csv"])
                        batch_btn = gr.Button("🚀 Run Batch Prediction",
                                              variant="primary", size="lg")
                    
                    with gr.Column():
                        batch_status = gr.Markdown("*Upload a CSV to get predictions*")
                        batch_results = gr.Dataframe(
                            label="Batch Predictions",
                            headers=["Route", "Airline", "Delay Probability",
                                      "Prediction", "Risk"]
                        )
                
                def make_sample():
                    path = create_sample_csv()
                    return gr.File(value=path, visible=True)
                
                sample_btn.click(fn=make_sample, outputs=[sample_file])
                batch_btn.click(
                    fn=predict_batch_ui,
                    inputs=[upload],
                    outputs=[batch_status, batch_results]
                )
            
            # ── TAB 4: ARCHITECTURE ───────────────────────────────────────
            with gr.TabItem("🏗️ Architecture", id="arch"):
                gr.Markdown("""
## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  BTS Raw CSV / Synthetic Generator                          │
│         ↓                                                   │
│  ETL Pipeline (etl/ingest.py + etl/clean.py)               │
│  • Load CSV → Validate → Clean → Create target label        │
│         ↓                                                   │
│  Feature Store (features/engineer.py)                       │
│  • 12 features: temporal, geographic, categorical           │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML LAYER                                  │
│  Training Pipeline (models/train.py)                        │
│  • Logistic Regression → baseline                           │
│  • Random Forest       → ensemble                           │
│  • XGBoost             → gradient boosting                  │
│  • LightGBM            → fast gradient boosting             │
│         ↓                                                   │
│  MLflow Tracking (mlflow_tracking/)                         │
│  • Parameters, metrics, model artifacts                     │
│         ↓                                                   │
│  Best Model Selection → Save pipeline.pkl                   │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                  SERVING LAYER                               │
│  FastAPI (api/main.py)                                      │
│  • POST /predict        → single prediction                 │
│  • POST /predict/batch  → batch prediction                  │
│  • GET  /model/info     → model metadata                    │
│  • GET  /health         → health check                      │
│         ↓                                                   │
│  Gradio UI (app.py)                                         │
│  • Interactive demo on Hugging Face Spaces                  │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                  MLOPS LAYER                                 │
│  Monitoring (monitoring/monitor.py)                         │
│  • Inference logging → data/inference_logs.csv              │
│  • Prediction drift detection                               │
│  • Evidently AI reports                                     │
│         ↓                                                   │
│  CI/CD (.github/workflows/deploy.yml)                       │
│  • Lint → Test → Build Docker → Push to ECR                 │
│         ↓                                                   │
│  Cloud Deployment                                           │
│  • AWS EC2 + ECR + RDS  OR  GCP Cloud Run                  │
└─────────────────────────────────────────────────────────────┘
```

## Features Engineered

| Feature | Type | Description |
|---------|------|-------------|
| DEP_DELAY | Numeric | Departure delay in minutes |
| DISTANCE | Numeric | Flight distance in miles |
| HOUR | Numeric | Departure hour (0-23) |
| DAY_OF_WEEK | Categorical | 1=Mon to 7=Sun |
| MONTH | Categorical | 1=Jan to 12=Dec |
| DIST_BUCKET | Categorical | Distance bucket (0-5) |
| IS_WEEKEND | Binary | Saturday or Sunday |
| IS_PEAK_HOUR | Binary | Rush hour (6-9 or 16-20) |
| SEASON | Categorical | 0=Winter, 1=Spring, 2=Summer, 3=Fall |
| AIRLINE_ENC | Encoded | Label-encoded airline |
| ORIGIN_ENC | Encoded | Label-encoded origin airport |
| DEST_ENC | Encoded | Label-encoded destination |

## Model Benchmarks (Expected)

| Model | ROC-AUC | F1 | Recall |
|-------|---------|-----|--------|
| Logistic Regression | ~0.72 | ~0.58 | ~0.62 |
| Random Forest | ~0.80 | ~0.68 | ~0.65 |
| XGBoost | ~0.84 | ~0.73 | ~0.70 |
| LightGBM | ~0.84 | ~0.73 | ~0.71 |

## Project Structure

```
flight_delay_prediction/
├── app.py                    ← Hugging Face Spaces entry point
├── api/
│   └── main.py               ← FastAPI REST API
├── etl/
│   ├── ingest.py             ← Data loading & synthetic generation
│   └── clean.py              ← Cleaning & validation
├── features/
│   └── engineer.py           ← Feature engineering pipeline
├── models/
│   ├── train.py              ← Multi-model training + MLflow
│   └── evaluate.py           ← Evaluation plots & reports
├── monitoring/
│   └── monitor.py            ← Drift detection & logging
├── tests/
│   └── test_pipeline.py      ← Unit tests
├── .github/workflows/
│   └── deploy.yml            ← CI/CD pipeline
├── Dockerfile                ← Container definition
├── requirements.txt          ← Dependencies
└── config.yaml               ← Configuration
```
                """)
            
            # ── TAB 5: API DOCS ───────────────────────────────────────────
            with gr.TabItem("📡 API Reference", id="api"):
                gr.Markdown("""
## REST API Reference

The FastAPI backend runs alongside this Gradio app.

### Base URL
```
http://localhost:7860  (local)
https://your-space.hf.space  (Hugging Face)
```

---

### `POST /predict` — Single Flight Prediction

**Request:**
```json
{
  "airline": "AA",
  "origin": "JFK",
  "destination": "LAX",
  "departure_delay": 15,
  "distance": 2475,
  "day_of_week": 5,
  "month": 7,
  "hour": 14
}
```

**Response:**
```json
{
  "prediction": "Delayed",
  "delay_probability": 0.7823,
  "confidence": "HIGH",
  "risk_level": "VERY HIGH",
  "model_name": "XGBoost",
  "inference_id": "INF-000042",
  "timestamp": "2024-01-15T14:32:11"
}
```

---

### `POST /predict/batch` — Batch Prediction (max 100)

**Request:**
```json
{
  "flights": [
    {"airline": "AA", "origin": "JFK", "destination": "LAX",
     "departure_delay": 0, "distance": 2475, "day_of_week": 3,
     "month": 6, "hour": 10},
    {"airline": "DL", "origin": "ATL", "destination": "ORD",
     "departure_delay": 20, "distance": 718, "day_of_week": 5,
     "month": 8, "hour": 17}
  ]
}
```

---

### `GET /health` — Health Check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoost",
  "total_predictions": 1247,
  "uptime": "running"
}
```

---

### `GET /model/info` — Model Information
```json
{
  "model_name": "XGBoost",
  "metrics": {
    "roc_auc": 0.8412,
    "f1": 0.7334,
    "precision": 0.7821,
    "recall": 0.6912
  },
  "features": ["DEP_DELAY", "DISTANCE", "HOUR", ...],
  "airlines": ["AA", "AS", "B6", ...],
  "airports": ["ATL", "BOS", "CLT", ...]
}
```

---

### `GET /inference/logs` — Recent Predictions
Returns last N inference log entries with drift analysis.

---

## curl Examples

```bash
# Single prediction
curl -X POST "http://localhost:7860/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"airline":"AA","origin":"JFK","destination":"LAX",
       "departure_delay":20,"distance":2475,
       "day_of_week":5,"month":7,"hour":17}'

# Health check
curl http://localhost:7860/health

# Model info
curl http://localhost:7860/model/info
```
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:20px; color:#94a3b8; font-size:0.85rem; border-top:1px solid #e2e8f0; margin-top:20px;">
            <strong>✈️ Flight Delay Prediction Platform</strong> · 
            Built with scikit-learn, XGBoost, LightGBM, MLflow, FastAPI & Gradio ·
            Data from <a href="https://transtats.bts.gov" target="_blank" style="color:#3b82f6;">BTS On-Time Performance Dataset</a>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    app_ui = build_ui()
    app_ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
