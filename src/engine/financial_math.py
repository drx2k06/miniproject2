"""Financial mathematics and amortization calculations for credit decisioning."""
from typing import Dict, List


def calculate_monthly_emi(
    principal: float,
    annual_rate_pct: float,
    tenure_months: int,
) -> float:
    """Calculate the Equated Monthly Installment (EMI) using reducing balance method.

    Formula:
        EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
        where:
            P = Principal loan amount
            r = Monthly interest rate (annual_rate_pct / 12 / 100)
            n = Loan tenure in months
    """
    if principal <= 0:
        return 0.0
    if tenure_months <= 0:
        raise ValueError("Loan tenure must be at least 1 month.")

    monthly_rate = (annual_rate_pct / 12.0) / 100.0

    # Special case: zero interest loan
    if abs(monthly_rate) < 1e-9:
        return round(principal / tenure_months, 2)

    factor = (1.0 + monthly_rate) ** tenure_months
    emi = principal * monthly_rate * factor / (factor - 1.0)
    return round(float(emi), 2)


def calculate_dti(
    monthly_income: float,
    existing_monthly_emis: float,
    proposed_emi: float,
) -> float:
    """Calculate the Debt-to-Income (DTI) ratio.

    DTI = (Existing Debt Payments + Proposed EMI) / Net Monthly Income
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be strictly positive.")
    if existing_monthly_emis < 0 or proposed_emi < 0:
        raise ValueError("Debt payments cannot be negative.")

    total_debt = existing_monthly_emis + proposed_emi
    return round(float(total_debt / monthly_income), 4)


def calculate_foir(
    monthly_income: float,
    existing_monthly_emis: float,
    proposed_emi: float,
) -> float:
    """Calculate Fixed Obligation to Income Ratio (FOIR) as a percentage."""
    return round(calculate_dti(monthly_income, existing_monthly_emis, proposed_emi) * 100.0, 2)


def calculate_max_eligible_loan(
    monthly_income: float,
    existing_monthly_emis: float,
    annual_rate_pct: float,
    tenure_months: int,
    target_dti: float = 0.45,
) -> float:
    """Calculate the maximum permissible loan amount that keeps DTI at or below target_dti."""
    if monthly_income <= 0:
        return 0.0

    allowed_total_debt = monthly_income * target_dti
    max_available_emi = allowed_total_debt - existing_monthly_emis

    if max_available_emi <= 0:
        return 0.0

    monthly_rate = (annual_rate_pct / 12.0) / 100.0
    if abs(monthly_rate) < 1e-9:
        return round(max_available_emi * tenure_months, 2)

    factor = (1.0 + monthly_rate) ** tenure_months
    max_principal = max_available_emi * (factor - 1.0) / (monthly_rate * factor)
    return round(float(max_principal), 2)


def generate_amortization_schedule(
    principal: float,
    annual_rate_pct: float,
    tenure_months: int,
) -> List[Dict[str, float]]:
    """Generate a complete month-by-month loan amortization schedule.

    Returns a list of dicts with:
        - month
        - opening_balance
        - emi
        - principal_paid
        - interest_paid
        - closing_balance
        - cumulative_interest
    """
    if principal <= 0 or tenure_months <= 0:
        return []

    emi = calculate_monthly_emi(principal, annual_rate_pct, tenure_months)
    monthly_rate = (annual_rate_pct / 12.0) / 100.0

    schedule = []
    balance = principal
    cumulative_interest = 0.0

    for month in range(1, tenure_months + 1):
        interest_for_month = balance * monthly_rate
        # In the final month, reconcile any rounding cents
        if month == tenure_months:
            principal_for_month = balance
            actual_emi = principal_for_month + interest_for_month
            closing_balance = 0.0
        else:
            principal_for_month = emi - interest_for_month
            if principal_for_month > balance:
                principal_for_month = balance
            closing_balance = balance - principal_for_month
            actual_emi = emi

        cumulative_interest += interest_for_month

        schedule.append({
            "month": month,
            "opening_balance": round(balance, 2),
            "emi": round(actual_emi, 2),
            "principal_paid": round(principal_for_month, 2),
            "interest_paid": round(interest_for_month, 2),
            "closing_balance": round(max(0.0, closing_balance), 2),
            "cumulative_interest": round(cumulative_interest, 2),
        })

        balance = closing_balance
        if balance <= 0:
            break

    return schedule
