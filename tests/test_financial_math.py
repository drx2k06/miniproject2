"""Unit tests for financial mathematics, EMI, and amortization schedules."""
import pytest
from src.engine.financial_math import (
    calculate_monthly_emi,
    calculate_dti,
    calculate_foir,
    calculate_max_eligible_loan,
    generate_amortization_schedule,
)


def test_calculate_monthly_emi_standard():
    """Verify EMI calculation against commercial banking benchmark."""
    # Principal: 100,000, Rate: 12% p.a., Tenure: 12 months -> EMI ~ 8884.88
    emi = calculate_monthly_emi(principal=100000.0, annual_rate_pct=12.0, tenure_months=12)
    assert pytest.approx(emi, rel=1e-3) == 8884.88


def test_calculate_monthly_emi_zero_interest():
    """Zero interest rate should cleanly yield principal / tenure."""
    emi = calculate_monthly_emi(principal=120000.0, annual_rate_pct=0.0, tenure_months=12)
    assert emi == 10000.0


def test_calculate_monthly_emi_zero_principal():
    """Zero principal should yield zero EMI."""
    emi = calculate_monthly_emi(principal=0.0, annual_rate_pct=10.0, tenure_months=24)
    assert emi == 0.0


def test_calculate_monthly_emi_invalid_tenure():
    """Non-positive tenure must raise ValueError."""
    with pytest.raises(ValueError, match="at least 1 month"):
        calculate_monthly_emi(principal=50000.0, annual_rate_pct=10.0, tenure_months=0)


def test_calculate_dti_standard():
    """Verify Debt-To-Income computation."""
    # Income: 100,000, Existing EMI: 20,000, Proposed EMI: 25,000 -> DTI = 0.45 (45%)
    dti = calculate_dti(monthly_income=100000.0, existing_monthly_emis=20000.0, proposed_emi=25000.0)
    assert dti == 0.45


def test_calculate_dti_invalid_inputs():
    """Negative debt or zero income must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_dti(monthly_income=0.0, existing_monthly_emis=5000.0, proposed_emi=5000.0)
    with pytest.raises(ValueError):
        calculate_dti(monthly_income=50000.0, existing_monthly_emis=-1000.0, proposed_emi=5000.0)


def test_calculate_foir():
    """FOIR should match DTI * 100."""
    foir = calculate_foir(monthly_income=80000.0, existing_monthly_emis=16000.0, proposed_emi=16000.0)
    assert foir == 40.0


def test_calculate_max_eligible_loan_inversion():
    """The maximum eligible loan should generate an EMI that exactly respects the target DTI."""
    income = 100000.0
    existing = 15000.0
    target_dti = 0.45
    rate = 10.5
    tenure = 36

    max_loan = calculate_max_eligible_loan(
        monthly_income=income,
        existing_monthly_emis=existing,
        annual_rate_pct=rate,
        tenure_months=tenure,
        target_dti=target_dti,
    )
    assert max_loan > 0.0

    # Test that the EMI of this max loan + existing debts brings DTI to exactly target_dti
    resulting_emi = calculate_monthly_emi(max_loan, rate, tenure)
    resulting_dti = calculate_dti(income, existing, resulting_emi)
    assert pytest.approx(resulting_dti, abs=0.01) == target_dti


def test_generate_amortization_schedule():
    """Check that amortization schedule terminates with zero balance."""
    principal = 200000.0
    rate = 11.0
    tenure = 24
    schedule = generate_amortization_schedule(principal, rate, tenure)

    assert len(schedule) == tenure
    assert schedule[0]["month"] == 1
    assert schedule[-1]["month"] == 24
    assert schedule[-1]["closing_balance"] == 0.0

    # Total principal paid across all months should equal initial principal
    total_principal_paid = sum(row["principal_paid"] for row in schedule)
    assert pytest.approx(total_principal_paid, abs=1.0) == principal
