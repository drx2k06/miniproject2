"""Domain engine for financial calculations, rules, and adverse action."""
from .financial_math import (
    calculate_monthly_emi,
    calculate_dti,
    calculate_foir,
    generate_amortization_schedule,
    calculate_max_eligible_loan,
)
from .rules_engine import UnderwritingRulesEngine
from .adverse_action import generate_adverse_action_advice

__all__ = [
    "calculate_monthly_emi",
    "calculate_dti",
    "calculate_foir",
    "generate_amortization_schedule",
    "calculate_max_eligible_loan",
    "UnderwritingRulesEngine",
    "generate_adverse_action_advice",
]
