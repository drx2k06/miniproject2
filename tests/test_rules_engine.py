"""
Unit Tests for Underwriting Policy Rules Engine and Regulatory Guardrails.
"""

import pytest
from src.data.schema import ApplicantProfile, EmploymentType, LoanPurpose, DecisionStatus
from src.engine.rules_engine import UnderwritingRulesEngine


def test_rules_engine_prime_applicant():
    engine = UnderwritingRulesEngine()
    applicant = ApplicantProfile(
        cibil_score=780,
        monthly_income=120000.0,
        loan_amount=500000.0,
        loan_tenure_months=36,
        existing_monthly_emis=15000.0,
        interest_rate_annual=10.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    metrics = engine.compute_metrics(applicant)
    flags, rationale, hard_decline = engine.evaluate_policy_rules(applicant, metrics)

    assert not hard_decline
    status, tier = engine.determine_decision(applicant, metrics, model_prob=0.85)
    assert status == DecisionStatus.APPROVED


def test_rules_engine_low_cibil_hard_stop():
    engine = UnderwritingRulesEngine(min_cibil_threshold=550)
    applicant = ApplicantProfile(
        cibil_score=510,  # Below 550 cutoff
        monthly_income=80000.0,
        loan_amount=300000.0,
        loan_tenure_months=24,
        existing_monthly_emis=5000.0,
        interest_rate_annual=14.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    metrics = engine.compute_metrics(applicant)
    flags, rationale, hard_decline = engine.evaluate_policy_rules(applicant, metrics)

    assert hard_decline
    status, tier = engine.determine_decision(applicant, metrics, model_prob=0.80)
    assert status == DecisionStatus.DECLINED
    assert any("CIBIL score" in f for f in flags)


def test_rules_engine_excessive_dti_hard_stop():
    engine = UnderwritingRulesEngine(max_dti_hard_cap=0.65)
    applicant = ApplicantProfile(
        cibil_score=720,
        monthly_income=40000.0,
        loan_amount=1200000.0,  # Huge loan for 40k income
        loan_tenure_months=24,
        existing_monthly_emis=15000.0,
        interest_rate_annual=12.0,
        employment_type=EmploymentType.SALARIED,
        loan_purpose=LoanPurpose.PERSONAL,
    )
    metrics = engine.compute_metrics(applicant)
    flags, rationale, hard_decline = engine.evaluate_policy_rules(applicant, metrics)

    assert hard_decline
    status, tier = engine.determine_decision(applicant, metrics, model_prob=0.80)
    assert status == DecisionStatus.DECLINED
    assert any("Debt-to-Income" in f for f in flags)
