"""
Unit Tests for Synthetic Credit Dataset Generator.
"""

import pytest
import pandas as pd
from src.data.generator import generate_synthetic_credit_dataset


def test_generator_structure_and_distributions():
    n = 200
    df = generate_synthetic_credit_dataset(n_samples=n, random_seed=42)

    assert len(df) == n
    expected_cols = [
        "age", "cibil_score", "monthly_income", "loan_amount",
        "loan_tenure_months", "existing_monthly_emis", "interest_rate_annual",
        "employment_type", "loan_purpose", "proposed_emi", "dti_ratio", "eligible",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    assert (df["cibil_score"] >= 300).all() and (df["cibil_score"] <= 900).all()
    assert (df["monthly_income"] > 0).all()
    assert (df["loan_amount"] > 0).all()
    assert set(df["eligible"].unique()).issubset({0, 1})


def test_generator_reproducibility():
    df1 = generate_synthetic_credit_dataset(n_samples=50, random_seed=99)
    df2 = generate_synthetic_credit_dataset(n_samples=50, random_seed=99)
    pd.testing.assert_frame_equal(df1, df2)
