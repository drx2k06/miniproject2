"""Data generation and schema definitions for CrediEdge AI."""
from .schema import (
    EmploymentType,
    LoanPurpose,
    RiskTier,
    DecisionStatus,
    ApplicantProfile,
    FinancialMetrics,
    CounterfactualAdvice,
    UnderwritingDecision,
)
from .generator import generate_synthetic_credit_dataset

__all__ = [
    "EmploymentType",
    "LoanPurpose",
    "RiskTier",
    "DecisionStatus",
    "ApplicantProfile",
    "FinancialMetrics",
    "CounterfactualAdvice",
    "UnderwritingDecision",
    "generate_synthetic_credit_dataset",
]
