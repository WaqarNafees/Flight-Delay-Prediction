"""
Unit Tests for Flight Delay Prediction Pipeline
Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Small raw DataFrame for testing."""
    from etl.ingest import generate_synthetic_data
    return generate_synthetic_data(n_samples=500, seed=42)


@pytest.fixture
def cleaned_df(raw_df):
    """Cleaned DataFrame."""
    from etl.clean import clean
    return clean(raw_df)


@pytest.fixture
def feature_df(cleaned_df):
    """Engineered feature DataFrame."""
    from features.engineer import engineer_features, encode_categoricals
    df = engineer_features(cleaned_df)
    df, _ = encode_categoricals(df, fit=True)
    return df


# ── ETL Tests ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_generate_synthetic_data_shape(self, raw_df):
        assert len(raw_df) == 500
        assert 'ARR_DELAY' in raw_df.columns
        assert 'DEP_DELAY' in raw_df.columns

    def test_synthetic_data_has_required_columns(self, raw_df):
        required = ['OP_CARRIER', 'ORIGIN', 'DEST', 'DISTANCE',
                     'DAY_OF_WEEK', 'ARR_DELAY', 'DEP_DELAY']
        for col in required:
            assert col in raw_df.columns, f"Missing: {col}"

    def test_delay_rate_reasonable(self, raw_df):
        delay_rate = (raw_df['ARR_DELAY'] > 15).mean()
        assert 0.1 <= delay_rate <= 0.5, f"Unusual delay rate: {delay_rate:.1%}"

    def test_distances_positive(self, raw_df):
        assert (raw_df['DISTANCE'] > 0).all()


class TestClean:
    def test_clean_creates_target(self, cleaned_df):
        assert 'is_delayed' in cleaned_df.columns

    def test_target_is_binary(self, cleaned_df):
        assert set(cleaned_df['is_delayed'].unique()).issubset({0, 1})

    def test_no_null_arr_delay(self, cleaned_df):
        assert cleaned_df['ARR_DELAY'].isna().sum() == 0

    def test_delay_threshold(self, cleaned_df):
        # Rows with ARR_DELAY > 15 should be labeled 1
        mask = cleaned_df['ARR_DELAY'] > 15
        assert (cleaned_df.loc[mask, 'is_delayed'] == 1).all()
        # Rows with ARR_DELAY <= 15 should be labeled 0
        mask2 = cleaned_df['ARR_DELAY'] <= 15
        assert (cleaned_df.loc[mask2, 'is_delayed'] == 0).all()

    def test_month_column_exists(self, cleaned_df):
        assert 'MONTH' in cleaned_df.columns
        assert cleaned_df['MONTH'].between(1, 12).all()

    def test_validate_passes(self, cleaned_df):
        from etl.clean import validate_dataframe
        valid, msg = validate_dataframe(cleaned_df)
        assert valid, msg


# ── Feature Engineering Tests ──────────────────────────────────────────────

class TestFeatureEngineering:
    def test_engineer_creates_dist_bucket(self, feature_df):
        assert 'DIST_BUCKET' in feature_df.columns

    def test_engineer_creates_is_weekend(self, feature_df):
        assert 'IS_WEEKEND' in feature_df.columns
        assert feature_df['IS_WEEKEND'].isin([0, 1]).all()

    def test_engineer_creates_season(self, feature_df):
        assert 'SEASON' in feature_df.columns
        assert feature_df['SEASON'].isin([0, 1, 2, 3]).all()

    def test_encoding_creates_airline_enc(self, feature_df):
        assert 'AIRLINE_ENC' in feature_df.columns
        assert feature_df['AIRLINE_ENC'].dtype in [int, np.int64, np.int32]

    def test_get_feature_matrix(self, cleaned_df):
        from features.engineer import build_full_feature_pipeline
        X, y, _ = build_full_feature_pipeline(cleaned_df)
        assert len(X) == len(y)
        assert X.shape[1] > 0
        assert y.nunique() == 2

    def test_no_nulls_in_features(self, cleaned_df):
        from features.engineer import build_full_feature_pipeline
        X, y, _ = build_full_feature_pipeline(cleaned_df)
        assert X.isna().sum().sum() == 0, "Features contain NaN values"


# ── Model Tests ────────────────────────────────────────────────────────────

class TestModel:
    @pytest.fixture(scope="class")
    def trained_model(self, tmp_path_factory):
        """Train a small model for testing."""
        from etl.ingest import generate_synthetic_data
        from etl.clean import clean
        from features.engineer import build_full_feature_pipeline
        from models.train import train_all_models
        
        raw = generate_synthetic_data(n_samples=1000, seed=42)
        cleaned = clean(raw)
        X, y, _ = build_full_feature_pipeline(cleaned)
        results = train_all_models(X, y)
        return results

    def test_model_has_predict_proba(self, trained_model):
        model = trained_model['best_model']
        assert hasattr(model, 'predict_proba')

    def test_roc_auc_above_baseline(self, trained_model):
        best_auc = trained_model['best_metrics']['roc_auc']
        assert best_auc > 0.6, f"ROC-AUC too low: {best_auc}"

    def test_all_models_trained(self, trained_model):
        assert len(trained_model['all_results']) >= 2

    def test_best_model_is_selected(self, trained_model):
        best_name = trained_model['best_name']
        best_auc = trained_model['best_metrics']['roc_auc']
        for name, metrics in trained_model['all_results'].items():
            assert metrics['roc_auc'] <= best_auc + 1e-6


# ── API Tests ──────────────────────────────────────────────────────────────

class TestAPI:
    def test_flight_input_validation(self):
        from api.main import FlightInput
        flight = FlightInput(
            airline="aa",  # Should be uppercased
            origin="jfk",
            destination="lax",
            departure_delay=15.0,
            distance=2475,
            day_of_week=5,
            month=7,
            hour=14
        )
        assert flight.airline == "AA"
        assert flight.origin == "JFK"
        assert flight.destination == "LAX"

    def test_invalid_day_of_week(self):
        from api.main import FlightInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FlightInput(
                airline="AA", origin="JFK", destination="LAX",
                departure_delay=0, distance=1000,
                day_of_week=8,  # Invalid: must be 1-7
                month=6, hour=12
            )

    def test_invalid_month(self):
        from api.main import FlightInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FlightInput(
                airline="AA", origin="JFK", destination="LAX",
                departure_delay=0, distance=1000,
                day_of_week=3,
                month=13,  # Invalid
                hour=12
            )

    def test_get_risk_level(self):
        from api.main import get_risk_level
        assert get_risk_level(0.1) == "LOW"
        assert get_risk_level(0.35) == "MODERATE"
        assert get_risk_level(0.55) == "HIGH"
        assert get_risk_level(0.80) == "VERY HIGH"


# ── Monitoring Tests ───────────────────────────────────────────────────────

class TestMonitoring:
    def test_compute_stats_empty(self):
        from monitoring.monitor import compute_prediction_stats
        result = compute_prediction_stats(None)
        assert result == {}

    def test_compute_stats(self):
        from monitoring.monitor import compute_prediction_stats
        df = pd.DataFrame({
            'prediction': ['Delayed', 'On Time', 'Delayed', 'On Time', 'On Time'],
            'probability': [0.7, 0.3, 0.8, 0.2, 0.4],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        stats = compute_prediction_stats(df)
        assert stats['total_predictions'] == 5
        assert abs(stats['delay_rate'] - 0.4) < 0.01

    def test_drift_detection_insufficient_data(self):
        from monitoring.monitor import check_prediction_drift
        df = pd.DataFrame({
            'prediction': ['Delayed'] * 10,
            'probability': [0.8] * 10,
            'timestamp': pd.Timestamp.utcnow()
        })
        result = check_prediction_drift(df)
        assert result['drift_detected'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
