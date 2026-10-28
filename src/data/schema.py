"""Pydantic schemas and domain models for credit underwriting."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EmploymentType(str, Enum):
    SALARIED = "Salaried"
    SELF_EMPLOYED = "Self-Employed"
    BUSINESS = "Business"
    OTHER = "Other"


class LoanPurpose(str, Enum):
    PERSONAL = "Personal"
    HOME = "Home"
    AUTO = "Auto"
    EDUCATION = "Education"
    DEBT_CONSOLIDATION = "Debt Consolidation"


class RiskTier(str, Enum):
    LOW_RISK = "Low Risk (Tier 1)"
    MODERATE_RISK = "Moderate Risk (Tier 2)"
    HIGH_RISK = "High Risk (Tier 3)"


class DecisionStatus(str, Enum):
    APPROVED = "Approved"
    MANUAL_REVIEW = "Manual Review"
    DECLINED = "Declined"


class ApplicantProfile(BaseModel):
    """Input profile for an applicant seeking a loan."""
    applicant_id: Optional[str] = Field(default=None, description="Unique applicant identifier")
    age: int = Field(default=32, ge=18, le=75, description="Applicant age in years")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL / Credit bureau score (300-900)")
    monthly_income: float = Field(..., gt=0, description="Monthly net take-home income in INR")
    loan_amount: float = Field(..., gt=0, description="Requested loan principal in INR")
    loan_tenure_months: int = Field(default=36, ge=6, le=360, description="Loan term in months")
    existing_monthly_emis: float = Field(default=0.0, ge=0.0, description="Existing monthly debt obligations")
    interest_rate_annual: float = Field(default=10.5, ge=1.0, le=40.0, description="Annual interest rate percentage")
    employment_type: EmploymentType = Field(default=EmploymentType.SALARIED, description="Applicant employment status")
    loan_purpose: LoanPurpose = Field(default=LoanPurpose.PERSONAL, description="Stated purpose of requested loan")

    @field_validator("existing_monthly_emis")
    @classmethod
    def validate_existing_emis(cls, v: float, values) -> float:
        if v < 0:
            raise ValueError("Existing monthly EMIs cannot be negative.")
        return float(v)


class FinancialMetrics(BaseModel):
    """Calculated financial and debt serviceability metrics."""
    proposed_emi: float = Field(..., description="Calculated monthly EMI for the requested loan")
    total_monthly_debt: float = Field(..., description="Existing EMIs + proposed EMI")
    debt_to_income_ratio: float = Field(..., description="DTI ratio (total monthly debt / monthly income)")
    foir_percentage: float = Field(..., description="Fixed Obligation to Income Ratio (%)")
    net_disposable_income: float = Field(..., description="Monthly income remaining after all debt payments")
    loan_to_annual_income: float = Field(..., description="Ratio of requested loan to applicant's annual income")
    monthly_income: float = Field(default=0.0, description="Applicant's monthly net income")
    loan_amount: float = Field(default=0.0, description="Principal requested loan amount")
    loan_tenure_months: int = Field(default=36, description="Loan term in months")

    @property
    def dti_ratio(self) -> float:
        return self.debt_to_income_ratio

    @property
    def foir(self) -> float:
        return round(self.foir_percentage / 100.0, 4)

    @property
    def annual_income(self) -> float:
        return round(self.monthly_income * 12.0, 2)

    @property
    def loan_to_income_multiple(self) -> float:
        return self.loan_to_annual_income

    @property
    def total_payment(self) -> float:
        return round(self.proposed_emi * self.loan_tenure_months, 2)

    @property
    def total_interest_payable(self) -> float:
        return round(max(0.0, self.total_payment - self.loan_amount), 2)


class CounterfactualAdvice(BaseModel):
    """Actionable prescriptive advice for declined or borderline applicants."""
    is_eligible: bool
    required_monthly_income: Optional[float] = None
    max_eligible_loan_amount: Optional[float] = None
    recommended_tenure_months: Optional[int] = None
    recommended_debt_paydown: Optional[float] = None
    advisory_notes: List[str] = Field(default_factory=list)

    @property
    def max_permissible_loan(self) -> float:
        return self.max_eligible_loan_amount if self.max_eligible_loan_amount is not None else 0.0

    @property
    def min_required_income(self) -> float:
        return self.required_monthly_income if self.required_monthly_income is not None else 0.0

    @property
    def required_tenure_extension_months(self) -> int:
        return self.recommended_tenure_months if self.recommended_tenure_months is not None else 0

    @property
    def required_debt_paydown(self) -> float:
        return self.recommended_debt_paydown if self.recommended_debt_paydown is not None else 0.0

    @property
    def recommended_actions(self) -> List[str]:
        return self.advisory_notes


class UnderwritingDecision(BaseModel):
    """Comprehensive decision output generated by CrediEdge AI."""
    applicant_id: Optional[str] = None
    status: DecisionStatus
    risk_tier: RiskTier
    approval_probability: float = Field(..., ge=0.0, le=1.0)
    default_probability: float = Field(..., ge=0.0, le=1.0)
    financial_metrics: FinancialMetrics
    policy_flags: List[str] = Field(default_factory=list)
    decision_rationale: List[str] = Field(default_factory=list)
    counterfactual_advice: Optional[CounterfactualAdvice] = None

    @property
    def metrics(self) -> FinancialMetrics:
        return self.financial_metrics

    @property
    def hard_stops_triggered(self) -> List[str]:
        return self.policy_flags

    @property
    def policy_warnings(self) -> List[str]:
        return self.policy_flags
