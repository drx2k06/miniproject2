"""Unit tests for ML pipelines, serialization, and batch inference."""
from pathlib import Path
import pytest
import pandas as pd
from src.data.generator import generate_synthetic_credit_dataset
from src.models.pipeline import CrediEdgePipeline, CrediEdgePredictor
from src.models.evaluation import compare_models


@pytest.fixture
def sample_dataset():
    """Generates small synthetic dataset for fast test runs."""
    return generate_synthetic_credit_dataset(n_samples=200, random_seed=42)


def test_pipeline_fit_and_predict(sample_dataset):
    """Verify that CrediEdgePipeline fits and predicts valid probabilities."""
    pipe = CrediEdgePipeline(model_type="logistic", random_state=42)
    pipe.fit(sample_dataset, sample_dataset["eligible"])

    probs = pipe.predict_proba(sample_dataset)
    assert probs.shape == (len(sample_dataset), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = pipe.predict(sample_dataset)
    assert len(preds) == len(sample_dataset)
    assert set(preds).issubset({0, 1})


def test_pipeline_serialization(sample_dataset, tmp_path):
    """Verify serialization to disk and deserialization parity."""
    pipe = CrediEdgePipeline(model_type="random_forest", random_state=42)
    pipe.fit(sample_dataset, sample_dataset["eligible"])

    save_path = tmp_path / "rf_model.joblib"
    pipe.save(save_path)
    assert save_path.exists()

    loaded_pipe = CrediEdgePipeline.load(save_path)
    orig_probs = pipe.predict_proba(sample_dataset)
    loaded_probs = loaded_pipe.predict_proba(sample_dataset)

    assert pytest.approx(orig_probs, abs=1e-5) == loaded_probs


def test_compare_models_suite(sample_dataset):
    """Check that compare_models evaluates multiple architectures successfully."""
    results_df, results_dict, models = compare_models(sample_dataset, test_size=0.3, random_state=42)

    assert not results_df.empty
    assert "accuracy" in results_df.columns
    assert "roc_auc" in results_df.columns
    assert len(models) >= 3
