---
title: Flight Delay Prediction Platform
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - machine-learning
  - mlops
  - flight-delay
  - xgboost
  - lightgbm
  - fastapi
  - scikit-learn
short_description: End-to-End ML System - ETL → XGBoost → MLflow → API
---

# ✈️ Flight Delay Prediction Platform

**End-to-End ML System** — Data Engineering → Feature Engineering → MLflow → XGBoost/LightGBM → FastAPI → MLOps

## What This Does

This is a **complete production-grade ML system** built on the U.S. Bureau of Transportation Statistics (BTS) On-Time Performance Dataset. It predicts whether a flight will be delayed more than 15 minutes.

## Tabs

| Tab | Description |
|-----|-------------|
| 🏋️ **Train Model** | Run full pipeline: ETL → Feature Engineering → Train 4 models → MLflow tracking → Auto-select best |
| 🎯 **Make Prediction** | Single flight prediction with risk gauge visualization |
| 📋 **Batch Predict** | Upload CSV for bulk predictions |
| 🏗️ **Architecture** | System design, features, model benchmarks |
| 📡 **API Reference** | REST API docs with curl examples |

## Quick Start

1. Click **Train Model** tab
2. Set sample size (30,000 recommended)
3. Click **Start Training Pipeline**
4. Wait ~60 seconds for training to complete
5. Switch to **Make Prediction** tab
6. Fill in flight details and click **Predict Delay**

## ML Pipeline

```
Synthetic BTS Data → ETL Cleaning → 12 Features → 4 Models → MLflow → Best Model
```

**Models compared:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost ⭐ (usually wins)
- LightGBM ⭐ (usually wins)

**Expected ROC-AUC:** ~0.82-0.86 for XGBoost/LightGBM

## REST API

The FastAPI backend is embedded. After training:

```bash
# Single prediction
curl -X POST "https://your-space.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"airline":"AA","origin":"JFK","destination":"LAX",
       "departure_delay":20,"distance":2475,"day_of_week":5,"month":7,"hour":14}'
```

## Features Engineered

`DEP_DELAY` · `DISTANCE` · `HOUR` · `DAY_OF_WEEK` · `MONTH` · `DIST_BUCKET` · `IS_WEEKEND` · `IS_PEAK_HOUR` · `SEASON` · `AIRLINE_ENC` · `ORIGIN_ENC` · `DEST_ENC`

## Tech Stack

- **Data:** pandas, NumPy, SQLAlchemy
- **ML:** scikit-learn, XGBoost, LightGBM
- **MLOps:** MLflow experiment tracking
- **API:** FastAPI + Pydantic
- **UI:** Gradio
- **Monitoring:** Evidently AI (drift detection)
- **CI/CD:** GitHub Actions
- **Cloud:** Docker + AWS ECR/EC2 or GCP Cloud Run

## Project Structure

```
├── app.py                  ← Gradio UI (this file runs on HF Spaces)
├── api/main.py             ← FastAPI REST API
├── etl/ingest.py           ← Data loading & synthetic generation
├── etl/clean.py            ← ETL cleaning pipeline
├── features/engineer.py    ← Feature engineering
├── models/train.py         ← Multi-model training + MLflow
├── models/evaluate.py      ← Evaluation plots
├── monitoring/monitor.py   ← Drift detection
├── tests/test_pipeline.py  ← Unit tests (pytest)
├── Dockerfile              ← Container
└── .github/workflows/      ← CI/CD
```

## Data Source

[BTS On-Time Performance Dataset](https://transtats.bts.gov/) — This demo uses synthetic data statistically calibrated to match the real BTS dataset distributions.

## For Production Use

1. Replace `generate_synthetic_data()` in `etl/ingest.py` with actual BTS CSV download
2. Add PostgreSQL connection string to `config.yaml`
3. Enable AWS deployment in `.github/workflows/deploy.yml`
4. Set up scheduled Airflow retraining DAG

## License

MIT
