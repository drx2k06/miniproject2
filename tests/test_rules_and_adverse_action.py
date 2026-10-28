"""Unit tests for underwriting policy rules, hard stops, and adverse action."""
import pytest
from src.data.schema import (
    ApplicantProfile,
    EmploymentType,
    LoanPurpose,
    DecisionStatus,
    RiskTier,
)
from src.engine.rules_engine import UnderwritingRulesEngine
from src.models.pipeline import CrediEdgePredictor


@pytest.fixture
def rules_engine():
    return UnderwritingRulesEngine()


@pytest.fixture
def predictor(rules_engine):
    return CrediEdgePredictor(rules_engine=rules_engine)


def test_subprime_cibil_hard_stop(predictor):
    """CIBIL score below 550 must trigger an immediate DECLINE and hard stop."""
    app = ApplicantProfile(
        cibil_score=490,  # Below minimum threshold of 550
        monthly_income=120000.0,
        loan_amount=200000.0,
        loan_tenure_months=36,
        existing_monthly_emis=5000.0,
        interest_rate_annual=11.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    decision = predictor.predict(app)

    assert decision.status == DecisionStatus.DECLINED
    assert decision.risk_tier == RiskTier.HIGH_RISK
    assert any("CIBIL score" in flag and "minimum threshold" in flag for flag in decision.policy_flags)


def test_excessive_dti_hard_stop(predictor):
    """DTI ratio exceeding 65% must trigger an immediate DECLINE."""
    app = ApplicantProfile(
        cibil_score=780,  # Prime score
        monthly_income=50000.0,
        loan_amount=2500000.0,  # Huge loan for 50k income
        loan_tenure_months=24,
        existing_monthly_emis=15000.0,
        interest_rate_annual=12.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    decision = predictor.predict(app)

    assert decision.status == DecisionStatus.DECLINED
    assert decision.risk_tier == RiskTier.HIGH_RISK
    assert decision.financial_metrics.debt_to_income_ratio > 0.65
    assert any("Debt-to-Income ratio" in flag and "exceeds maximum limit" in flag for flag in decision.policy_flags)


def test_prime_applicant_auto_approval(predictor):
    """Prime applicant with high CIBIL and very low DTI should be APPROVED."""
    app = ApplicantProfile(
        cibil_score=810,
        monthly_income=150000.0,
        loan_amount=400000.0,
        loan_tenure_months=36,
        existing_monthly_emis=10000.0,
        interest_rate_annual=9.5,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    decision = predictor.predict(app)

    assert decision.status == DecisionStatus.APPROVED
    assert decision.risk_tier == RiskTier.LOW_RISK
    assert decision.financial_metrics.debt_to_income_ratio <= 0.45
    assert decision.counterfactual_advice is None


def test_adverse_action_counterfactual_recommendations(predictor):
    """A declined applicant due to high DTI must receive actionable advice."""
    app = ApplicantProfile(
        cibil_score=710,
        monthly_income=60000.0,
        loan_amount=1500000.0,
        loan_tenure_months=24,
        existing_monthly_emis=10000.0,
        interest_rate_annual=11.5,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    decision = predictor.predict(app)

    assert decision.status in [DecisionStatus.DECLINED, DecisionStatus.MANUAL_REVIEW]
    assert decision.counterfactual_advice is not None
    cf = decision.counterfactual_advice

    # Recommended permissible loan must be strictly less than requested loan
    assert cf.max_permissible_loan < app.loan_amount
    assert cf.min_required_income > app.monthly_income
    assert len(cf.advisory_notes) >= 2
