"""Policy rules engine and regulatory guardrails for credit underwriting."""
from typing import List, Tuple
from ..data.schema import (
    ApplicantProfile,
    FinancialMetrics,
    DecisionStatus,
    RiskTier,
)
from .financial_math import (
    calculate_monthly_emi,
    calculate_dti,
    calculate_foir,
)


class UnderwritingRulesEngine:
    """Enforces banking compliance policies, hard stops, and risk tier assignments."""

    def __init__(
        self,
        min_cibil_threshold: int = 550,
        prime_cibil_threshold: int = 720,
        max_dti_hard_cap: float = 0.65,
        target_dti_cap: float = 0.45,
        max_lti_personal: float = 12.0,
        max_lti_home: float = 60.0,
    ):
        self.min_cibil_threshold = min_cibil_threshold
        self.prime_cibil_threshold = prime_cibil_threshold
        self.max_dti_hard_cap = max_dti_hard_cap
        self.target_dti_cap = target_dti_cap
        self.max_lti_personal = max_lti_personal
        self.max_lti_home = max_lti_home

    def compute_metrics(self, applicant: ApplicantProfile) -> FinancialMetrics:
        """Derive standard financial serviceability metrics."""
        proposed_emi = calculate_monthly_emi(
            applicant.loan_amount,
            applicant.interest_rate_annual,
            applicant.loan_tenure_months,
        )
        total_debt = applicant.existing_monthly_emis + proposed_emi
        dti = calculate_dti(
            applicant.monthly_income,
            applicant.existing_monthly_emis,
            proposed_emi,
        )
        foir = calculate_foir(
            applicant.monthly_income,
            applicant.existing_monthly_emis,
            proposed_emi,
        )
        net_disposable = applicant.monthly_income - total_debt
        annual_income = applicant.monthly_income * 12.0
        lti = applicant.loan_amount / annual_income if annual_income > 0 else 0.0

        return FinancialMetrics(
            proposed_emi=round(proposed_emi, 2),
            total_monthly_debt=round(total_debt, 2),
            debt_to_income_ratio=round(dti, 4),
            foir_percentage=round(foir, 2),
            net_disposable_income=round(net_disposable, 2),
            loan_to_annual_income=round(lti, 2),
            monthly_income=applicant.monthly_income,
            loan_amount=applicant.loan_amount,
            loan_tenure_months=applicant.loan_tenure_months,
        )

    def evaluate_policy_rules(
        self,
        applicant: ApplicantProfile,
        metrics: FinancialMetrics,
    ) -> Tuple[List[str], List[str], bool]:
        """Check underwriting policy rules and hard stops.

        Returns:
            - policy_flags: Warnings and critical notices
            - decision_rationale: Explanatory reasons
            - is_hard_declined: True if a hard stop is violated
        """
        flags: List[str] = []
        rationale: List[str] = []
        hard_decline = False

        # Rule 1: Minimum CIBIL Score Cutoff
        if applicant.cibil_score < self.min_cibil_threshold:
            flags.append(f"CRITICAL: CIBIL score ({applicant.cibil_score}) is below minimum threshold ({self.min_cibil_threshold}).")
            rationale.append("Severe credit risk due to subprime credit bureau history.")
            hard_decline = True
        elif applicant.cibil_score < 650:
            flags.append(f"WARNING: Sub-optimal credit score ({applicant.cibil_score}) requires compensatory factors.")

        # Rule 2: Maximum DTI Hard Cap
        if metrics.debt_to_income_ratio > self.max_dti_hard_cap:
            flags.append(f"CRITICAL: Debt-to-Income ratio (DTI ratio: {metrics.debt_to_income_ratio:.1%}) exceeds maximum limit ({self.max_dti_hard_cap:.1%}).")
            rationale.append("High debt service obligation increases likelihood of default.")
            hard_decline = True
        elif metrics.debt_to_income_ratio > self.target_dti_cap:
            flags.append(f"WARNING: Debt-to-Income ratio (DTI ratio: {metrics.debt_to_income_ratio:.1%}) exceeds target standard ({self.target_dti_cap:.1%}).")

        # Rule 3: Existing Obligations Stress
        existing_ratio = applicant.existing_monthly_emis / applicant.monthly_income
        if existing_ratio > 0.50:
            flags.append(f"WARNING: Existing obligations alone consume {existing_ratio:.1%} of monthly income.")

        # Rule 4: Loan to Income Ratio
        max_allowed_lti = self.max_lti_home if applicant.loan_purpose == "Home" else self.max_lti_personal
        if metrics.loan_to_annual_income > max_allowed_lti:
            flags.append(f"POLICY: Requested loan is {metrics.loan_to_annual_income:.1f}x annual income (cap is {max_allowed_lti:.1f}x).")
            rationale.append("Requested loan amount is disproportionate to demonstrated annual earning capacity.")

        # Positive rationales
        if applicant.cibil_score >= self.prime_cibil_threshold:
            rationale.append(f"Prime credit bureau score ({applicant.cibil_score}) demonstrates strong repayment discipline.")
        if metrics.debt_to_income_ratio <= self.target_dti_cap:
            rationale.append(f"Healthy DTI ratio ({metrics.debt_to_income_ratio:.1%}) indicates comfortable debt service capacity.")

        return flags, rationale, hard_decline

    def evaluate_rules(
        self,
        applicant: ApplicantProfile,
        metrics: FinancialMetrics,
    ) -> Tuple[DecisionStatus, List[str], List[str]]:
        """Evaluates policy rules and returns (status, hard_stops, warnings)."""
        flags, rationale, hard_decline = self.evaluate_policy_rules(applicant, metrics)
        hard_stops = [f for f in flags if "CRITICAL" in f]
        warnings = [f for f in flags if "WARNING" in f or "POLICY" in f]

        if hard_decline:
            status = DecisionStatus.DECLINED
        elif warnings:
            status = DecisionStatus.MANUAL_REVIEW
        else:
            status = DecisionStatus.APPROVED

        return status, hard_stops, warnings

    def determine_decision(
        self,
        applicant: ApplicantProfile,
        metrics: FinancialMetrics,
        model_prob: float,
    ) -> Tuple[DecisionStatus, RiskTier]:
        """Harmonize statistical model probability with policy rules to produce final decision."""
        _, _, hard_decline = self.evaluate_policy_rules(applicant, metrics)

        if hard_decline:
            return DecisionStatus.DECLINED, RiskTier.HIGH_RISK

        # Underwriting composite decisioning
        if model_prob >= 0.70 and applicant.cibil_score >= 700 and metrics.debt_to_income_ratio <= self.target_dti_cap:
            return DecisionStatus.APPROVED, RiskTier.LOW_RISK

        if model_prob < 0.40 or applicant.cibil_score < 620 or metrics.debt_to_income_ratio > 0.55:
            return DecisionStatus.DECLINED, RiskTier.HIGH_RISK

        # Borderline applicants route to underwriter review
        return DecisionStatus.MANUAL_REVIEW, RiskTier.MODERATE_RISK
