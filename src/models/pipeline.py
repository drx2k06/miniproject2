"""Production Machine Learning Pipelines and Unified Inference Predictor for CrediEdge AI."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.schema import (
    ApplicantProfile,
    DecisionStatus,
    FinancialMetrics,
    RiskTier,
    UnderwritingDecision,
)
from ..engine.adverse_action import generate_adverse_action_advice
from ..engine.rules_engine import UnderwritingRulesEngine
from .scratch_logistic import ScratchLogisticRegression


NUMERIC_FEATURES = [
    "cibil_score",
    "monthly_income",
    "loan_amount",
    "loan_tenure_months",
    "existing_monthly_emis",
    "interest_rate_annual",
    "proposed_emi",
    "dti_ratio",
]

CATEGORICAL_FEATURES = [
    "employment_type",
    "loan_purpose",
]

BASELINE_2D_FEATURES = [
    "cibil_score",
    "loan_amount",
]


class CrediEdgePipeline:
    """Production ML pipeline factory and wrapper."""

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None

    @staticmethod
    def create_preprocessor() -> ColumnTransformer:
        """Constructs robust ColumnTransformer for numerical and categorical features."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ]
        )

    @classmethod
    def create_model_suite(cls, random_state: int = 42) -> Dict[str, Pipeline]:
        """Creates complete benchmark suite of underwriting classifiers."""
        suite = {
            "logistic_regression": Pipeline([
                ("preprocessor", cls.create_preprocessor()),
                ("classifier", LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                )),
            ]),
            "random_forest": Pipeline([
                ("preprocessor", cls.create_preprocessor()),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]),
            "gradient_boosting": Pipeline([
                ("preprocessor", cls.create_preprocessor()),
                ("classifier", HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=5,
                    random_state=random_state,
                )),
            ]),
            "scratch_baseline_2d": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", ScratchLogisticRegression(
                    l2_reg=0.05,
                    solver="bfgs",
                    random_state=random_state,
                )),
            ]),
        }
        return suite

    def _build_pipeline(self) -> Pipeline:
        preprocessor = self.create_preprocessor()

        if self.model_type == "logistic":
            classifier = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif self.model_type in ["rf", "random_forest"]:
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type in ["gb", "gradient_boosting"]:
            classifier = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}.")

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "CrediEdgePipeline":
        """Fit the preprocessing and classifier pipeline on training data."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities: col 0 = P(Declined), col 1 = P(Eligible)."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet.")
        return self.pipeline.predict_proba(X[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predict binary eligibility labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CrediEdgePipeline":
        """Load serialized pipeline from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        return joblib.load(path)

    @staticmethod
    def save_artifact(obj: Any, filepath: Union[str, Path]) -> None:
        """Serializes trained pipeline artifact via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)

    @staticmethod
    def load_artifact(filepath: Union[str, Path]) -> Any:
        """Loads serialized pipeline artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        return joblib.load(path)


class CrediEdgePredictor:
    """Unified inference engine integrating ML predictions with banking policy rules."""

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        model_pipeline: Optional[Any] = None,
        rules_engine: Optional[UnderwritingRulesEngine] = None,
    ):
        self.pipeline = pipeline or model_pipeline
        self.rules_engine = rules_engine or UnderwritingRulesEngine()

    def predict(self, applicant: ApplicantProfile) -> UnderwritingDecision:
        """Score an individual applicant profile and generate a complete decision."""
        # Compute financial serviceability metrics
        metrics = self.rules_engine.compute_metrics(applicant)

        # Statistical ML prediction
        prob_approval = 0.5
        prob_default = 0.5

        if self.pipeline is not None:
            # Check if self.pipeline is CrediEdgePipeline wrapper or raw sklearn Pipeline
            pipe_obj = getattr(self.pipeline, "pipeline", self.pipeline)

            row_dict = {
                "cibil_score": [applicant.cibil_score],
                "monthly_income": [applicant.monthly_income],
                "loan_amount": [applicant.loan_amount],
                "loan_tenure_months": [applicant.loan_tenure_months],
                "existing_monthly_emis": [applicant.existing_monthly_emis],
                "interest_rate_annual": [applicant.interest_rate_annual],
                "proposed_emi": [metrics.proposed_emi],
                "dti_ratio": [metrics.debt_to_income_ratio],
                "employment_type": [applicant.employment_type.value],
                "loan_purpose": [applicant.loan_purpose.value],
            }
            df_row = pd.DataFrame(row_dict)

            try:
                probs = pipe_obj.predict_proba(df_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
                prob_approval = float(probs[1])
                prob_default = float(probs[0])
            except Exception:
                try:
                    df_2d = df_row[BASELINE_2D_FEATURES].values
                    probs = pipe_obj.predict_proba(df_2d)[0]
                    prob_approval = float(probs[1])
                    prob_default = float(probs[0])
                except Exception:
                    cibil_norm = (applicant.cibil_score - 300) / 600.0
                    dti_penalty = min(1.0, max(0.0, (metrics.debt_to_income_ratio - 0.20) / 0.40))
                    prob_approval = round(float(np.clip(0.6 * cibil_norm + 0.4 * (1.0 - dti_penalty), 0.05, 0.95)), 4)
                    prob_default = round(1.0 - prob_approval, 4)
        else:
            cibil_norm = (applicant.cibil_score - 300) / 600.0
            dti_penalty = min(1.0, max(0.0, (metrics.debt_to_income_ratio - 0.20) / 0.40))
            prob_approval = round(float(np.clip(0.6 * cibil_norm + 0.4 * (1.0 - dti_penalty), 0.05, 0.95)), 4)
            prob_default = round(1.0 - prob_approval, 4)

        # Policy rules check
        policy_flags, rationale, hard_decline = self.rules_engine.evaluate_policy_rules(applicant, metrics)
        status, risk_tier = self.rules_engine.determine_decision(applicant, metrics, prob_approval)

        # Prescriptive counterfactual advice for declined/review cases
        counterfactual = generate_adverse_action_advice(
            applicant=applicant,
            metrics=metrics,
            status=status,
            target_dti=self.rules_engine.target_dti_cap,
        )

        return UnderwritingDecision(
            applicant_id=applicant.applicant_id,
            status=status,
            risk_tier=risk_tier,
            approval_probability=round(prob_approval, 4),
            default_probability=round(prob_default, 4),
            financial_metrics=metrics,
            policy_flags=policy_flags,
            decision_rationale=rationale,
            counterfactual_advice=counterfactual,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire DataFrame of applicants and return enriched results."""
        results = []
        for idx, row in df.iterrows():
            applicant = ApplicantProfile(
                applicant_id=str(row.get("applicant_id", f"APP-{idx+1:04d}")),
                age=int(row.get("age", 35)),
                cibil_score=int(row["cibil_score"]),
                monthly_income=float(row["monthly_income"]),
                loan_amount=float(row["loan_amount"]),
                loan_tenure_months=int(row.get("loan_tenure_months", 36)),
                existing_monthly_emis=float(row.get("existing_monthly_emis", 0.0)),
                interest_rate_annual=float(row.get("interest_rate_annual", 10.5)),
                employment_type=str(row.get("employment_type", "Salaried")),
                loan_purpose=str(row.get("loan_purpose", "Personal")),
            )
            decision = self.predict(applicant)
            results.append({
                "applicant_id": decision.applicant_id,
                "cibil_score": applicant.cibil_score,
                "monthly_income": applicant.monthly_income,
                "loan_amount": applicant.loan_amount,
                "tenure_months": applicant.loan_tenure_months,
                "proposed_emi": decision.financial_metrics.proposed_emi,
                "dti_ratio": decision.financial_metrics.debt_to_income_ratio,
                "approval_probability": decision.approval_probability,
                "decision_status": decision.status.value,
                "risk_tier": decision.risk_tier.value,
                "policy_flags_count": len(decision.policy_flags),
                "advisory_notes": " | ".join(decision.counterfactual_advice.advisory_notes) if decision.counterfactual_advice else "N/A (Approved)",
            })

        return pd.DataFrame(results)
