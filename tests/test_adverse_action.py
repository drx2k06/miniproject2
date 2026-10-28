"""
Unit Tests for Adverse Action and Counterfactual Advice Generation.
"""

import pytest
from src.data.schema import ApplicantProfile, EmploymentType, LoanPurpose, DecisionStatus
from src.engine.rules_engine import UnderwritingRulesEngine
from src.engine.adverse_action import generate_adverse_action_advice


def test_adverse_action_approved_returns_none():
    applicant = ApplicantProfile(
        cibil_score=800,
        monthly_income=150000.0,
        loan_amount=400000.0,
        loan_tenure_months=36,
        existing_monthly_emis=10000.0,
        interest_rate_annual=9.5,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    engine = UnderwritingRulesEngine()
    metrics = engine.compute_metrics(applicant)
    advice = generate_adverse_action_advice(applicant, metrics, status=DecisionStatus.APPROVED)
    assert advice is None


def test_adverse_action_declined_generates_prescriptive_advice():
    applicant = ApplicantProfile(
        cibil_score=680,
        monthly_income=50000.0,
        loan_amount=1500000.0,  # High loan request
        loan_tenure_months=24,
        existing_monthly_emis=15000.0,
        interest_rate_annual=12.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    engine = UnderwritingRulesEngine()
    metrics = engine.compute_metrics(applicant)
    advice = generate_adverse_action_advice(applicant, metrics, status=DecisionStatus.DECLINED)

    assert advice is not None
    assert advice.max_permissible_loan < applicant.loan_amount
    assert advice.min_required_income > applicant.monthly_income
    assert len(advice.recommended_actions) >= 1
