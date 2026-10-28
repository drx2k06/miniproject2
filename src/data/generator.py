"""Realistic credit underwriting dataset synthesizer for CrediEdge AI."""
from typing import Optional
import numpy as np
import pandas as pd


def generate_synthetic_credit_dataset(
    n_samples: int = 3000,
    random_seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a realistic retail credit risk dataset reflecting commercial banking distributions.

    Features:
        - age: Applicant age (21 - 65)
        - cibil_score: Credit score (300 - 900)
        - monthly_income: Net monthly earnings in INR
        - loan_amount: Requested loan principal in INR
        - loan_tenure_months: Repayment duration in months
        - existing_monthly_emis: Current monthly debt obligations in INR
        - interest_rate_annual: Annual interest rate (%)
        - employment_type: Salaried, Self-Employed, Business
        - loan_purpose: Personal, Home, Auto, Education, Debt Consolidation
        - dti_ratio: Debt-to-Income ratio (derived)
        - eligible: Ground-truth binary approval label (0 = Declined, 1 = Eligible)
    """
    rng = np.random.default_rng(random_seed)

    # 1. Age: truncated normal between 21 and 65
    age = np.clip(rng.normal(loc=35, scale=9, size=n_samples).astype(int), 21, 65)

    # 2. Employment status
    emp_choices = ["Salaried", "Self-Employed", "Business"]
    emp_probs = [0.65, 0.25, 0.10]
    employment_type = rng.choice(emp_choices, size=n_samples, p=emp_probs)

    # 3. Monthly Income: log-normal distribution with median ~ ₹55,000
    income_log = rng.normal(loc=10.9, scale=0.6, size=n_samples)
    monthly_income = np.clip(np.exp(income_log), 18000, 500000).round(-2)

    # 4. CIBIL score: bimodal mixture modeling prime borrowers and subprime borrowers
    prime_scores = rng.normal(loc=740, scale=45, size=n_samples)
    subprime_scores = rng.normal(loc=580, scale=65, size=n_samples)
    mask = rng.random(size=n_samples) < 0.70  # 70% prime
    cibil_score = np.where(mask, prime_scores, subprime_scores)
    cibil_score = np.clip(cibil_score.round(), 300, 900).astype(int)

    # 5. Loan purpose
    purpose_choices = ["Personal", "Home", "Auto", "Education", "Debt Consolidation"]
    purpose_probs = [0.40, 0.25, 0.20, 0.10, 0.05]
    loan_purpose = rng.choice(purpose_choices, size=n_samples, p=purpose_probs)

    # 6. Loan Tenure: depends on loan purpose
    tenure_months = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        purp = loan_purpose[i]
        if purp == "Home":
            tenure_months[i] = rng.choice([60, 120, 180, 240, 300])
        elif purp == "Auto":
            tenure_months[i] = rng.choice([24, 36, 48, 60, 84])
        elif purp == "Education":
            tenure_months[i] = rng.choice([36, 60, 84, 120])
        else:
            tenure_months[i] = rng.choice([12, 24, 36, 48, 60])

    # 7. Loan Amount: scaled by monthly income and loan purpose
    amount_multiplier = np.zeros(n_samples)
    for i in range(n_samples):
        purp = loan_purpose[i]
        if purp == "Home":
            mult = rng.uniform(20.0, 60.0)  # Home loans are larger
        elif purp == "Auto":
            mult = rng.uniform(6.0, 18.0)
        elif purp == "Education":
            mult = rng.uniform(4.0, 15.0)
        else:
            mult = rng.uniform(2.0, 10.0)  # Personal / Debt consolidation
        amount_multiplier[i] = mult

    loan_amount = (monthly_income * amount_multiplier * rng.uniform(0.85, 1.15)).round(-3)
    loan_amount = np.clip(loan_amount, 25000, 15000000)

    # 8. Existing Monthly EMIs: currently serviced debt
    existing_emi_ratio = rng.beta(a=1.5, b=5.0, size=n_samples)  # typically 0% to 35% of income
    existing_monthly_emis = (monthly_income * existing_emi_ratio).round(-2)

    # 9. Interest Rate: risk-adjusted based on CIBIL and purpose
    base_rate = np.zeros(n_samples)
    for i in range(n_samples):
        score = cibil_score[i]
        if score >= 750:
            rate = rng.uniform(8.5, 11.0)
        elif score >= 650:
            rate = rng.uniform(11.0, 14.5)
        else:
            rate = rng.uniform(14.5, 20.0)
        base_rate[i] = round(rate, 2)
    interest_rate_annual = base_rate

    # 10. Compute monthly EMI for requested loan using reducing balance formula
    monthly_rates = (interest_rate_annual / 12.0) / 100.0
    factor = np.power(1.0 + monthly_rates, tenure_months)
    proposed_emi = loan_amount * monthly_rates * factor / (factor - 1.0)

    # 11. Calculate Debt-To-Income (DTI) ratio
    total_monthly_debt = existing_monthly_emis + proposed_emi
    dti_ratio = total_monthly_debt / monthly_income

    # 12. Realistic ground-truth latent eligibility score
    # High CIBIL increases approval; High DTI severely penalizes approval
    cibil_standardized = (cibil_score - 650) / 100.0
    dti_penalty = (dti_ratio - 0.45) * 5.0
    income_bonus = np.log10(monthly_income / 30000) * 0.8
    emp_bonus = np.where(employment_type == "Salaried", 0.3, np.where(employment_type == "Business", 0.1, -0.1))

    latent_score = (
        0.5
        + 1.8 * cibil_standardized
        - 2.4 * dti_penalty
        + 0.5 * income_bonus
        + emp_bonus
        + rng.normal(0, 0.4, size=n_samples)  # Real-world stochastic noise
    )

    # Hard regulatory vetoes (Underwriting policy boundaries):
    # - CIBIL < 550 or DTI > 0.65 are strict declines in banking practice
    prob_approval = 1.0 / (1.0 + np.exp(-latent_score))
    prob_approval = np.where(cibil_score < 550, 0.02, prob_approval)
    prob_approval = np.where(dti_ratio > 0.65, 0.03, prob_approval)

    eligible = (prob_approval >= 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "cibil_score": cibil_score,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "loan_tenure_months": tenure_months,
        "existing_monthly_emis": existing_monthly_emis,
        "interest_rate_annual": interest_rate_annual,
        "employment_type": employment_type,
        "loan_purpose": loan_purpose,
        "proposed_emi": proposed_emi.round(2),
        "dti_ratio": dti_ratio.round(4),
        "eligible": eligible,
    })

    if output_path:
        df.to_csv(output_path, index=False)

    return df
