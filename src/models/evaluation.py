"""
Model Evaluation, Cross-Validation, and Benchmarking Suite for CrediEdge AI.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
)

from .pipeline import (
    CrediEdgePipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BASELINE_2D_FEATURES,
)


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Calculates comprehensive out-of-sample evaluation metrics."""
    y_true = np.asarray(y_test, dtype=int)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = 0.5
    brier = float(brier_score_loss(y_true, y_proba))
    loss = float(log_loss(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "confusion_matrix": cm,
    }


def compare_models(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Executes head-to-head benchmarking comparing the original 2-feature prototype
    against multi-factor regularized logistic regression and ensemble models.
    """
    target_col = "eligible"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[target_col].values

    # Stratified train/test split to preserve approval ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    suite = CrediEdgePipeline.create_model_suite(random_state=random_state)
    results_list: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, pipeline in suite.items():
        if name == "scratch_baseline_2d":
            # Baseline only uses CIBIL score and Loan Amount (mirroring original repository)
            X_tr_sub = X_train[BASELINE_2D_FEATURES].values
            X_te_sub = X_test[BASELINE_2D_FEATURES].values
            pipeline.fit(X_tr_sub, y_train)
            metrics = evaluate_classifier(pipeline, X_te_sub, y_test, model_name="Baseline 2D (CIBIL+Loan)")
        else:
            pipeline.fit(X_train, y_train)
            display_name = name.replace("_", " ").title()
            metrics = evaluate_classifier(pipeline, X_test, y_test, model_name=display_name)

        results_list.append(metrics)
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results_list)

    # Save artifacts if requested
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for name, pipeline in trained_models.items():
            CrediEdgePipeline.save_artifact(pipeline, artifacts_dir / f"{name}.joblib")

        # Save metrics JSON
        with open(artifacts_dir / "evaluation_results.json", "w") as f:
            json.dump(results_list, f, indent=2)

    return results_df, {r["model_name"]: r for r in results_list}, trained_models
