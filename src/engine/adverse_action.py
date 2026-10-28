"""Adverse Action and Counterfactual Advice Generation for rejected/borderline applicants."""
from typing import List, Optional
from ..data.schema import (
    ApplicantProfile,
    FinancialMetrics,
    DecisionStatus,
    CounterfactualAdvice,
)
from .financial_math import (
    calculate_monthly_emi,
    calculate_max_eligible_loan,
)


def generate_adverse_action_advice(
    applicant: ApplicantProfile,
    metrics: FinancialMetrics,
    status: DecisionStatus,
    target_dti: float = 0.45,
    max_tenure_allowed: int = 84,
) -> Optional[CounterfactualAdvice]:
    """Compute prescriptive counterfactual guidance to help applicants achieve loan approval.

    Calculates:
        1. Max permissible loan amount at current income and obligations.
        2. Minimum required income to service current loan request.
        3. Tenure extension needed to reduce monthly EMI.
        4. Debt paydown needed on existing monthly obligations.
    """
    if status == DecisionStatus.APPROVED:
        return None

    notes: List[str] = []

    # 1. Calculate maximum eligible loan amount at target DTI
    max_loan = calculate_max_eligible_loan(
        monthly_income=applicant.monthly_income,
        existing_monthly_emis=applicant.existing_monthly_emis,
        annual_rate_pct=applicant.interest_rate_annual,
        tenure_months=applicant.loan_tenure_months,
        target_dti=target_dti,
    )

    if max_loan > 0 and max_loan < applicant.loan_amount:
        notes.append(
            f"Reduce requested loan amount from ₹{applicant.loan_amount:,.0f} to ₹{max_loan:,.0f} "
            f"to meet the {target_dti:.0%} DTI threshold."
        )
    elif max_loan <= 0:
        notes.append(
            f"Existing monthly debt obligations (₹{applicant.existing_monthly_emis:,.0f}) consume "
            f"{applicant.existing_monthly_emis / applicant.monthly_income:.1%} of income, leaving no room for new borrowing."
        )

    # 2. Calculate required monthly income for requested loan amount
    # Target: (existing_emis + proposed_emi) / required_income = target_dti
    required_income = (applicant.existing_monthly_emis + metrics.proposed_emi) / target_dti
    if required_income > applicant.monthly_income:
        notes.append(
            f"Increase verifiable monthly net income to at least ₹{required_income:,.0f} "
            f"(e.g., via co-borrower / co-applicant addition)."
        )

    # 3. Explore tenure extension to lower EMI
    recommended_tenure: Optional[int] = None
    if applicant.loan_tenure_months < max_tenure_allowed:
        # Check if extending tenure can achieve target DTI
        allowed_emi = (applicant.monthly_income * target_dti) - applicant.existing_monthly_emis
        if allowed_emi > 0:
            for test_tenure in range(applicant.loan_tenure_months + 6, max_tenure_allowed + 1, 6):
                test_emi = calculate_monthly_emi(
                    applicant.loan_amount,
                    applicant.interest_rate_annual,
                    test_tenure,
                )
                if test_emi <= allowed_emi:
                    recommended_tenure = test_tenure
                    notes.append(
                        f"Extend loan tenure from {applicant.loan_tenure_months} months to {recommended_tenure} months "
                        f"to reduce monthly EMI from ₹{metrics.proposed_emi:,.0f} to ₹{test_emi:,.0f}."
                    )
                    break

    # 4. Debt Paydown Recommendation
    recommended_paydown: Optional[float] = None
    if applicant.existing_monthly_emis > 0 and metrics.debt_to_income_ratio > target_dti:
        allowed_total_debt = applicant.monthly_income * target_dti
        excess_debt = metrics.total_monthly_debt - allowed_total_debt
        if excess_debt > 0 and excess_debt <= applicant.existing_monthly_emis:
            recommended_paydown = excess_debt
            notes.append(
                f"Pay off existing loans/credit cards to reduce monthly debt obligations by ₹{excess_debt:,.0f}/month."
            )

    # 5. CIBIL score rehabilitation advice if sub-optimal
    if applicant.cibil_score < 650:
        notes.append(
            f"Rehabilitate CIBIL score (currently {applicant.cibil_score}) by maintaining zero 30+ DPD defaults "
            "for 6 consecutive months and keeping revolving credit card utilization below 30%."
        )

    return CounterfactualAdvice(
        is_eligible=False,
        required_monthly_income=round(required_income, 2) if required_income else None,
        max_eligible_loan_amount=round(max_loan, 2) if max_loan > 0 else 0.0,
        recommended_tenure_months=recommended_tenure,
        recommended_debt_paydown=round(recommended_paydown, 2) if recommended_paydown else None,
        advisory_notes=notes,
    )
