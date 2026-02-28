"""
✈️ Flight Delay Prediction — Streamlit App
Full ML Pipeline: ETL → Feature Engineering → Train → Predict
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
import streamlit as st

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #bfdbfe; margin: 5px 0 0; font-size: 1rem; }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; color: #1e40af; }
    .metric-card p  { font-size: 0.85rem; color: #64748b; margin: 4px 0 0; }

    .result-delayed {
        background: #fef2f2; border: 2px solid #dc2626;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .result-ontime {
        background: #f0fdf4; border: 2px solid #16a34a;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .result-delayed h2 { color: #dc2626; font-size: 1.8rem; margin: 0; }
    .result-ontime  h2 { color: #16a34a; font-size: 1.8rem; margin: 0; }

    .tag {
        display: inline-block;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin: 3px;
    }
    .tag-blue   { background: #dbeafe; color: #1e40af; }
    .tag-green  { background: #dcfce7; color: #166534; }
    .tag-yellow { background: #fef3c7; color: #92400e; }
    .tag-purple { background: #f3e8ff; color: #6b21a8; }

    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
AIRLINES = {
    "AA": "American Airlines",   "DL": "Delta Air Lines",
    "UA": "United Airlines",     "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",     "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",     "F9": "Frontier Airlines",
    "G4": "Allegiant Air",       "HA": "Hawaiian Airlines",
    "9E": "Endeavor Air",        "OO": "SkyWest Airlines",
    "MQ": "Envoy Air",           "OH": "PSA Airlines",
    "YX": "Republic Airways",    "YV": "Mesa Airlines"
}
AIRPORTS = sorted([
    "ATL","ORD","DFW","DEN","LAX","CLT","LAS","PHX","MCO","SEA",
    "MSP","DTW","BOS","EWR","JFK","SFO","LGA","FLL","IAD","IAH",
    "MIA","MDW","PHL","BWI","SLC","PDX","HOU","OAK","AUS","SAN","TPA"
])
DAYS   = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",
           5:"Friday",6:"Saturday",7:"Sunday"}
MONTHS = {1:"January",2:"February",3:"March",4:"April",5:"May",
           6:"June",7:"July",8:"August",9:"September",
           10:"October",11:"November",12:"December"}

MODEL_PATH   = "models/pipeline.pkl"
ENCODER_PATH = "models/encoders.pkl"
METRICS_PATH = "models/metrics.json"

# ── Session State ──────────────────────────────────────────────────────────
if "model"          not in st.session_state: st.session_state.model          = None
if "encoders"       not in st.session_state: st.session_state.encoders       = None
if "model_info"     not in st.session_state: st.session_state.model_info     = {}
if "train_results"  not in st.session_state: st.session_state.train_results  = None
if "trained"        not in st.session_state: st.session_state.trained        = False
if "inference_logs" not in st.session_state: st.session_state.inference_logs = []

# ── Load existing model on startup ────────────────────────────────────────
def try_load_model():
    import joblib
    if os.path.exists(MODEL_PATH) and st.session_state.model is None:
        st.session_state.model    = joblib.load(MODEL_PATH)
        st.session_state.encoders = joblib.load(ENCODER_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                st.session_state.model_info = json.load(f)
        st.session_state.trained = True

try_load_model()

# ── Helpers ────────────────────────────────────────────────────────────────
def get_risk(prob):
    if prob < 0.30: return "🟢 LOW",       "#16a34a"
    if prob < 0.50: return "🟡 MODERATE",  "#ca8a04"
    if prob < 0.70: return "🟠 HIGH",      "#ea580c"
    return              "🔴 VERY HIGH",    "#dc2626"

def get_confidence(prob):
    m = abs(prob - 0.5)
    if m > 0.35: return "HIGH"
    if m > 0.20: return "MEDIUM"
    return "LOW"

def make_gauge(prob, title="Delay Probability"):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    segments = [(0,.25,'#16a34a','Low'),(0.25,.45,'#ca8a04','Moderate'),
                (.45,.65,'#ea580c','High'),(.65,1,'#dc2626','Very High')]
    for s, e, c, lbl in segments:
        ax.barh(0, e-s, left=s, height=0.55, color=c, alpha=0.85)
        ax.text((s+e)/2, -0.42, lbl, ha='center', fontsize=9,
                color='white', fontweight='bold')
    ax.axvline(prob, color='#0f172a', lw=3, ymin=0.05, ymax=0.95)
    ax.plot(prob, 0, 'v', color='#0f172a', ms=13)
    ax.text(prob, 0.5, f'{prob:.1%}', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#0f172a')
    ax.set_xlim(0,1); ax.set_ylim(-0.65, 0.85)
    ax.set_xlabel('Delay Probability', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈️ Flight Delay ML")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏋️  Train Model",
        "🎯  Single Prediction",
        "📋  Batch Prediction",
        "📊  Model Metrics",
        "🏗️  Architecture"
    ])

    st.markdown("---")
    if st.session_state.trained:
        info = st.session_state.model_info
        name = info.get("model_name", "Unknown")
        auc  = info.get("metrics", {}).get("roc_auc", "N/A")
        st.success(f"✅ Model Ready\n\n**{name}**\n\nROC-AUC: **{auc}**")
    else:
        st.warning("⚠️ No model trained yet.\n\nGo to **Train Model** first.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#94a3b8;'>
    <b>Tech Stack</b><br>
    scikit-learn · XGBoost<br>
    LightGBM · MLflow<br>
    FastAPI · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>✈️ Flight Delay Prediction Platform</h1>
    <p>End-to-End ML System &nbsp;·&nbsp; ETL → Feature Engineering → MLflow → XGBoost/LightGBM → API</p>
    <div style='margin-top:10px;'>
        <span class='tag tag-blue'>📦 scikit-learn</span>
        <span class='tag tag-green'>🚀 XGBoost / LightGBM</span>
        <span class='tag tag-yellow'>📊 MLflow Tracking</span>
        <span class='tag tag-purple'>🔌 FastAPI Backend</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════
if "Train" in page:

    st.markdown("## 🏋️ Training Pipeline")
    st.markdown("Runs the full pipeline: load data → ETL → feature engineering → train 4 models → MLflow → save best.")

    col1, col2 = st.columns([1, 2])

    with col1:
        n_samples = st.slider("Training Samples", 5000, 50000, 20000, 5000,
                               help="More samples = better model, longer training time")
        st.markdown("""
        **Pipeline Steps:**
        1. 📂 Load bundled BTS flight data
        2. 🧹 ETL: clean & validate
        3. ⚙️ Feature engineering (12 features)
        4. 🤖 Train 4 models
        5. 📊 MLflow experiment tracking
        6. 🏆 Auto-select best by ROC-AUC
        7. 💾 Save full sklearn Pipeline
        """)
        train_btn = st.button("🚀 Start Training Pipeline", type="primary",
                               use_container_width=True)

    with col2:
        log_box = st.empty()

    if train_btn:
        logs = []

        def log(msg):
            logs.append(msg)
            log_box.code("\n".join(logs), language="bash")

        try:
            # Step 1
            log("📂 Step 1/5 — Loading flight data...")
            from etl.ingest import load_or_generate
            raw = load_or_generate(n_samples=n_samples)
            log(f"   ✅ Loaded {len(raw):,} flight records")

            # Step 2
            log("\n🧹 Step 2/5 — Cleaning data...")
            from etl.clean import clean, get_data_summary
            cleaned = clean(raw)
            s = get_data_summary(cleaned)
            log(f"   ✅ {s['total_flights']:,} flights retained")
            log(f"   Delay rate (>15min): {s['delay_rate']:.1%}")
            log(f"   Airlines: {s['airlines']}  |  Routes: {s['routes']}")

            # Step 3
            log("\n⚙️  Step 3/5 — Feature engineering...")
            from features.engineer import build_full_feature_pipeline
            X, y, encoders = build_full_feature_pipeline(cleaned)
            log(f"   ✅ {X.shape[1]} features × {len(X):,} samples")
            log(f"   Features: {', '.join(X.columns.tolist())}")

            # Step 4
            log("\n🤖 Step 4/5 — Training models (may take ~30-60s)...")
            from models.train import train_all_models, save_model
            with st.spinner("Training in progress..."):
                results = train_all_models(X, y)

            log("\n📊 Model Comparison:")
            for name, m in results['all_results'].items():
                flag = " ⭐ BEST" if name == results['best_name'] else ""
                log(f"   {name:22s}  AUC={m['roc_auc']:.4f}  "
                    f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}{flag}")

            # Step 5
            log(f"\n💾 Step 5/5 — Saving best model: {results['best_name']}...")
            save_model(results['best_model'],
                       metrics=results['best_metrics'],
                       model_name=results['best_name'])

            import joblib
            st.session_state.model      = joblib.load(MODEL_PATH)
            st.session_state.encoders   = joblib.load(ENCODER_PATH)
            with open(METRICS_PATH) as f:
                st.session_state.model_info = json.load(f)
            st.session_state.train_results = results
            st.session_state.trained = True

            log(f"\n✅ Training complete!")
            log(f"   Best model: {results['best_name']}")
            log(f"   ROC-AUC:    {results['best_metrics']['roc_auc']:.4f}")
            log(f"   F1-Score:   {results['best_metrics']['f1']:.4f}")
            log(f"\n→ Switch to 'Single Prediction' to predict delays!")

            st.success(f"🏆 Best Model: **{results['best_name']}** | ROC-AUC: **{results['best_metrics']['roc_auc']:.4f}**")
            st.balloons()

        except Exception as e:
            import traceback
            log(f"\n❌ Error: {str(e)}")
            log(traceback.format_exc())
            st.error(f"Training failed: {str(e)}")

    # Show evaluation plots if trained
    if st.session_state.train_results is not None:
        st.markdown("---")
        st.markdown("### 📊 Evaluation Results")

        results = st.session_state.train_results

        from models.evaluate import (plot_roc_curve, plot_confusion_matrix,
                                      plot_model_comparison, plot_feature_importance)
        from models.train import get_feature_importance
        from features.engineer import ALL_FEATURES

        c1, c2 = st.columns(2)
        with c1:
            fig = plot_roc_curve(results['best_model'], results['X_test'],
                                  results['y_test'], results['best_name'])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            fig = plot_confusion_matrix(results['best_model'], results['X_test'],
                                         results['y_test'], results['best_name'])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        c3, c4 = st.columns(2)
        with c3:
            fig = plot_model_comparison(results['all_results'])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c4:
            fi_df = get_feature_importance(results['best_model'], ALL_FEATURES)
            if not fi_df.empty:
                fig = plot_feature_importance(fi_df)
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif "Single" in page:

    st.markdown("## 🎯 Single Flight Prediction")
    st.markdown("Predict whether a specific flight will be delayed more than **15 minutes**.")

    if not st.session_state.trained:
        st.warning("⚠️ Please train the model first — go to **Train Model** in the sidebar.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ✈️ Flight Details")

        airline_str = st.selectbox(
            "Airline",
            [f"{code} — {name}" for code, name in AIRLINES.items()],
            index=0
        )

        c1, c2 = st.columns(2)
        with c1:
            origin = st.selectbox("🛫 Origin", AIRPORTS, index=AIRPORTS.index("JFK"))
        with c2:
            dest = st.selectbox("🛬 Destination", AIRPORTS, index=AIRPORTS.index("LAX"))

        distance = st.slider("📏 Distance (miles)", 50, 5000, 2475, 50)
        dep_delay = st.slider("⏱️ Departure Delay (min)", -30, 180, 0, 5,
                               help="Set to 0 if predicting before departure")

        c3, c4 = st.columns(2)
        with c3:
            day_str = st.selectbox("📅 Day of Week", list(DAYS.values()), index=4)
        with c4:
            month_str = st.selectbox("🗓️ Month", list(MONTHS.values()), index=6)

        hour = st.slider("🕐 Departure Hour", 0, 23, 14, 1)

        predict_btn = st.button("🔮 Predict Delay", type="primary",
                                 use_container_width=True)

    with col2:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            try:
                airline_code = airline_str.split(" — ")[0].strip()
                day_num   = [k for k, v in DAYS.items()   if v == day_str][0]
                month_num = [k for k, v in MONTHS.items() if v == month_str][0]

                record = {
                    "OP_CARRIER": airline_code,
                    "ORIGIN":     origin,
                    "DEST":       dest,
                    "DEP_DELAY":  float(dep_delay),
                    "DISTANCE":   float(distance),
                    "DAY_OF_WEEK": int(day_num),
                    "MONTH":       int(month_num),
                    "HOUR":        int(hour),
                    "DEP_TIME":    int(hour) * 100
                }

                from features.engineer import encode_single_record
                X = encode_single_record(record, st.session_state.encoders)
                prob = float(st.session_state.model.predict_proba(X)[0][1])
                pred = "Delayed 🔴" if prob >= 0.5 else "On Time 🟢"
                risk, risk_color = get_risk(prob)
                conf = get_confidence(prob)

                # Result card
                css_class = "result-delayed" if prob >= 0.5 else "result-ontime"
                st.markdown(f"""
                <div class="{css_class}">
                    <h2>{pred}</h2>
                    <p style='font-size:1.1rem; margin:8px 0 0; color:#374151;'>
                        Delay Probability: <strong>{prob:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Gauge
                fig = make_gauge(prob, f"{origin} → {dest}")
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", risk)
                m2.metric("Confidence", conf)
                m3.metric("Probability", f"{prob:.1%}")

                # Details table
                model_name = st.session_state.model_info.get("model_name", "Unknown")
                st.markdown(f"""
                | Detail | Value |
                |--------|-------|
                | Route | {origin} → {dest} |
                | Airline | {AIRLINES.get(airline_code, airline_code)} |
                | Distance | {distance:,} miles |
                | Dep. Delay | {dep_delay:+.0f} min |
                | Day | {day_str} |
                | Model | {model_name} |
                """)

                # Log it
                st.session_state.inference_logs.append({
                    "time":        datetime.datetime.now().strftime("%H:%M:%S"),
                    "route":       f"{origin}→{dest}",
                    "airline":     airline_code,
                    "prediction":  "Delayed" if prob >= 0.5 else "On Time",
                    "probability": f"{prob:.1%}"
                })

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

        else:
            st.info("👈 Fill in flight details and click **Predict Delay**")

        # Example flights
        st.markdown("---")
        st.markdown("### 🎲 Quick Examples")
        examples = [
            ("AA", "JFK", "LAX", 45, 2475, 5, 7, 17, "High delay risk"),
            ("DL", "ATL", "ORD",  0,  718, 2, 3,  8, "Low delay risk"),
            ("WN", "MDW", "DEN", -5,  920, 6,12, 14, "Weekend flight"),
            ("UA", "SFO", "EWR", 90, 2565, 1, 8,  7, "Very high risk"),
        ]
        for airline, org, dst, dd, dist, dow, mo, hr, label in examples:
            if st.button(f"{org}→{dst} ({label})", use_container_width=True):
                st.info(f"Set: {airline} | {org}→{dst} | Dep delay: {dd}min | {dist}mi")

# ══════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif "Batch" in page:

    st.markdown("## 📋 Batch Prediction")
    st.markdown("Upload a CSV file to predict delays for multiple flights at once.")

    if not st.session_state.trained:
        st.warning("⚠️ Please train the model first.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📄 CSV Format Required")
        st.code("""airline,origin,destination,departure_delay,distance,day_of_week,month,hour
AA,JFK,LAX,20,2475,5,7,14
DL,ATL,ORD,0,718,2,3,9
WN,MDW,DEN,-5,920,6,12,14""", language="csv")

        # Sample CSV download
        sample_df = pd.DataFrame([
            {"airline":"AA","origin":"JFK","destination":"LAX",
             "departure_delay":20,"distance":2475,"day_of_week":5,"month":7,"hour":14},
            {"airline":"DL","origin":"ATL","destination":"ORD",
             "departure_delay":0,"distance":718,"day_of_week":2,"month":3,"hour":9},
            {"airline":"WN","origin":"MDW","destination":"DEN",
             "departure_delay":-5,"distance":920,"day_of_week":6,"month":12,"hour":14},
            {"airline":"UA","origin":"SFO","destination":"EWR",
             "departure_delay":45,"distance":2565,"day_of_week":1,"month":8,"hour":7},
        ])
        csv_bytes = sample_df.to_csv(index=False).encode()
        st.download_button("📥 Download Sample CSV", csv_bytes,
                            "sample_flights.csv", "text/csv",
                            use_container_width=True)

        uploaded = st.file_uploader("📂 Upload your CSV", type=["csv"])
        run_batch = st.button("🚀 Run Batch Prediction", type="primary",
                               use_container_width=True, disabled=uploaded is None)

    with col2:
        if run_batch and uploaded:
            try:
                df = pd.read_csv(uploaded)
                required = ['airline','origin','destination','distance','day_of_week','month']
                missing  = [c for c in required if c not in df.columns]

                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    from features.engineer import encode_single_record
                    results = []
                    prog = st.progress(0)

                    for i, row in df.iterrows():
                        record = {
                            "OP_CARRIER": str(row.get('airline','AA')).upper(),
                            "ORIGIN":     str(row.get('origin','ATL')).upper(),
                            "DEST":       str(row.get('destination','ORD')).upper(),
                            "DEP_DELAY":  float(row.get('departure_delay', 0)),
                            "DISTANCE":   float(row.get('distance', 1000)),
                            "DAY_OF_WEEK": int(row.get('day_of_week', 3)),
                            "MONTH":       int(row.get('month', 6)),
                            "HOUR":        int(row.get('hour', 12)),
                            "DEP_TIME":    int(row.get('hour', 12)) * 100
                        }
                        X   = encode_single_record(record, st.session_state.encoders)
                        prob = float(st.session_state.model.predict_proba(X)[0][1])
                        results.append({
                            "Route":        f"{record['ORIGIN']}→{record['DEST']}",
                            "Airline":      record['OP_CARRIER'],
                            "Delay Prob":   f"{prob:.1%}",
                            "Prediction":   "🔴 Delayed" if prob >= 0.5 else "🟢 On Time",
                            "Risk":         get_risk(prob)[0]
                        })
                        prog.progress((i+1)/len(df))

                    result_df = pd.DataFrame(results)
                    delayed   = (result_df['Prediction'].str.contains('Delayed')).sum()
                    ontime    = len(result_df) - delayed

                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Flights",  len(result_df))
                    m2.metric("🔴 Delayed",     delayed)
                    m3.metric("🟢 On Time",     ontime)

                    st.dataframe(result_df, use_container_width=True, height=400)

                    # Download results
                    csv_out = result_df.to_csv(index=False).encode()
                    st.download_button("📥 Download Results CSV", csv_out,
                                       "predictions.csv", "text/csv",
                                       use_container_width=True)

                    # Pie chart
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie([delayed, ontime],
                            labels=['Delayed', 'On Time'],
                            colors=['#dc2626','#16a34a'],
                            autopct='%1.1f%%', startangle=90)
                    ax.set_title('Prediction Split')
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Upload a CSV file and click **Run Batch Prediction**")

            # Show recent inference logs
            if st.session_state.inference_logs:
                st.markdown("### 🕐 Recent Predictions")
                st.dataframe(pd.DataFrame(st.session_state.inference_logs),
                             use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════
elif "Metrics" in page:

    st.markdown("## 📊 Model Metrics & Monitoring")

    if not st.session_state.trained:
        st.warning("⚠️ Train the model first to see metrics.")
        st.stop()

    info = st.session_state.model_info
    metrics = info.get("metrics", {})
    model_name = info.get("model_name", "Unknown")

    st.markdown(f"### 🏆 Best Model: `{model_name}`")

    # Big metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC",   f"{metrics.get('roc_auc',  0):.4f}", "↑ higher is better")
    c2.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
    c3.metric("F1-Score",  f"{metrics.get('f1',       0):.4f}")
    c4.metric("Precision", f"{metrics.get('precision',0):.4f}")
    c5.metric("Recall",    f"{metrics.get('recall',   0):.4f}")

    st.markdown("---")

    # Benchmark comparison
    st.markdown("### 📈 Expected Benchmarks (BTS Dataset)")
    bench_df = pd.DataFrame({
        "Model":   ["Logistic Regression","Random Forest","XGBoost","LightGBM"],
        "ROC-AUC": ["~0.70–0.75","~0.78–0.82","~0.82–0.86","~0.82–0.86"],
        "Notes":   ["Baseline","Good ensemble","Usually best","Fast & accurate"]
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Inference logs
    st.markdown("### 🔍 Inference Log (This Session)")
    if st.session_state.inference_logs:
        log_df = pd.DataFrame(st.session_state.inference_logs)
        st.dataframe(log_df, use_container_width=True)

        delayed_rate = (log_df['prediction'] == 'Delayed').mean()
        st.metric("Session Delay Rate", f"{delayed_rate:.1%}",
                   help="% of predictions that were 'Delayed' this session")

        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#dc2626' if p == 'Delayed' else '#16a34a'
                   for p in log_df['prediction']]
        probs = [float(p.strip('%'))/100 for p in log_df['probability']]
        ax.bar(range(len(probs)), probs, color=colors, alpha=0.85)
        ax.axhline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold')
        ax.set_xlabel('Prediction #'); ax.set_ylabel('Delay Probability')
        ax.set_title('Session Predictions', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("No predictions made yet in this session.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════
elif "Architecture" in page:

    st.markdown("## 🏗️ System Architecture")

    st.code("""
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
│  BTS Raw CSV / Synthetic Generator (20K records)    │
│         ↓                                           │
│  ETL Pipeline  (etl/ingest.py + etl/clean.py)       │
│  • Load → Validate → Clean → Create target label    │
│         ↓                                           │
│  Feature Store  (features/engineer.py)              │
│  • 12 features: temporal, geographic, encoded       │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    ML LAYER                          │
│  Training Pipeline  (models/train.py)               │
│  • Logistic Regression  (baseline)                  │
│  • Random Forest        (ensemble)                  │
│  • XGBoost              (gradient boosting)         │
│  • LightGBM             (fast gradient boosting)    │
│         ↓                                           │
│  MLflow Tracking  (mlflow_tracking/)                │
│  • Parameters, metrics, model artifacts             │
│         ↓                                           │
│  Best Model Selection → Save pipeline.pkl           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                 SERVING LAYER                        │
│  FastAPI  (api/main.py)                             │
│  • POST /predict        → single prediction         │
│  • POST /predict/batch  → batch prediction          │
│  • GET  /health         → health check              │
│         ↓                                           │
│  Streamlit UI  (streamlit_app.py)                   │
│  • Interactive demo on Streamlit Cloud              │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                 MLOPS LAYER                          │
│  Monitoring  (monitoring/monitor.py)                │
│  • Inference logging & drift detection              │
│  CI/CD  (.github/workflows/deploy.yml)              │
│  • Lint → Test → Docker → AWS ECR/EC2               │
└─────────────────────────────────────────────────────┘
    """, language="text")

    st.markdown("### 📐 Features Engineered")
    feat_df = pd.DataFrame({
        "Feature":     ["DEP_DELAY","DISTANCE","HOUR","DAY_OF_WEEK","MONTH",
                         "DIST_BUCKET","IS_WEEKEND","IS_PEAK_HOUR","SEASON",
                         "AIRLINE_ENC","ORIGIN_ENC","DEST_ENC"],
        "Type":        ["Numeric","Numeric","Numeric","Categorical","Categorical",
                         "Categorical","Binary","Binary","Categorical",
                         "Encoded","Encoded","Encoded"],
        "Description": [
            "Departure delay in minutes",
            "Flight distance in miles",
            "Departure hour (0–23)",
            "Day of week (1=Mon … 7=Sun)",
            "Month (1–12)",
            "Distance bucket (0=short … 5=very long)",
            "1 if Saturday or Sunday",
            "1 if rush hour (6–9am or 4–8pm)",
            "0=Winter, 1=Spring, 2=Summer, 3=Fall",
            "Label-encoded airline carrier",
            "Label-encoded origin airport",
            "Label-encoded destination airport"
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("### 📁 Project Structure")
    st.code("""
flight_delay_prediction/
├── streamlit_app.py          ← Streamlit UI (this file)
├── app.py                    ← Gradio UI (for HuggingFace)
├── api/main.py               ← FastAPI REST API
├── etl/
│   ├── ingest.py             ← Data loading & synthetic generation
│   └── clean.py              ← ETL cleaning pipeline
├── features/
│   └── engineer.py           ← 12-feature engineering pipeline
├── models/
│   ├── train.py              ← Multi-model training + MLflow
│   ├── evaluate.py           ← Evaluation plots & reports
│   ├── pipeline.pkl          ← Saved model (after training)
│   └── encoders.pkl          ← Saved encoders
├── monitoring/
│   └── monitor.py            ← Drift detection & logging
├── data/
│   └── raw/flights.csv       ← 20K BTS-style flight records
├── tests/test_pipeline.py    ← Unit tests (pytest)
├── .github/workflows/        ← CI/CD pipeline
├── Dockerfile                ← Container for AWS/GCP
└── requirements.txt          ← Dependencies
    """, language="text")
