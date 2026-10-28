# CrediEdge AI — Model Evaluation & Performance Report

**Platform:** CrediEdge AI (Autonomous Credit Risk Underwriting Engine)  
**Lead Architect & Evaluator:** Daniel Richard Xavier  
**Evaluation Date:** October 28, 2026  
**Dataset Size:** 2,500 Synthesized Credit Records (Reflecting Commercial Retail Lending)  
**Holdout Test Partition:** 625 Samples (25.0% Stratified Holdout)  

---

## 1. Evaluation Methodology & Verification Framework

In retail banking credit risk modeling, evaluation must never rely solely on training accuracy. In accordance with Basel III and fair credit underwriting standards, CrediEdge AI benchmarks models using:

1. **Stratified Holdout Evaluation:** 25% of data is strictly partitioned out-of-sample before training, maintaining the exact class ratio (44.2% eligible, 55.8% declined).
2. **Multi-Metric Assessment:**
   - **Directional Accuracy:** Overall classification correctness.
   - **Precision & Recall:** Evaluating false positive approvals (credit default risk) versus false negative declines (lost lending revenue).
   - **F1-Score:** Harmonic balance between precision and recall.
   - **ROC-AUC (Area Under ROC Curve):** Discriminative ability across all possible classification probability thresholds.
   - **Brier Calibration Score:** Mean squared difference between predicted probabilities and actual outcomes ($0.0$ = perfect calibration).
   - **Log-Loss (Binary Cross-Entropy):** Uncertainty and probabilistic loss.
3. **Head-to-Head Comparison:** Direct benchmark against the original 2-feature prototype architecture (CIBIL Score + Loan Amount only).

---

## 2. Benchmark Evaluation Results

The holdout test set was evaluated across all four models trained under identical random seed conditions (`random_state=42`):

| Model Architecture | Features Used | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 2D (Original Concept)** | 2 (CIBIL, Loan Amount) | 71.36% | 0.6590 | 0.7283 | 0.6919 | 0.8054 | 0.1774 |
| **Multi-Factor Logistic Regression** | 10 (Full Profile + DTI) | 95.68% | 0.9462 | 0.9565 | 0.9514 | 0.9946 | 0.0302 |
| **Random Forest Classifier** | 10 (Full Profile + DTI) | 95.84% | 0.9340 | 0.9746 | 0.9539 | 0.9932 | 0.0376 |
| **Gradient Boosting (Champion)** | 10 (Full Profile + DTI) | **97.12%** | **0.9542** | **0.9819** | **0.9679** | **0.9971** | **0.0218** |

---

## 3. Key Findings & Quantitative Analysis

### 3.1. The Critical Debt Serviceability Gap (+25.76% Accuracy Gain)
- The original reference prototype (`Li.py` & `eligibilityy.py`) attempted to predict loan approval using only two features: `CIBIL` and `Loan Amount`.
- When tested on realistic applicant data, this 2-feature baseline achieves only **71.36% accuracy** and a high Brier score of **0.1774**.
- **Root Cause of Baseline Failure:** In credit underwriting, risk is fundamentally driven by *debt serviceability*. A borrower with an excellent credit score of 780 earning ₹30,000/month who requests a ₹20,00,000 personal loan is a severe default risk. The 2-feature baseline erroneously approves such applicants because it lacks income and DTI data.
- By introducing income, tenure, existing monthly obligations, and proposed EMI, CrediEdge AI's Multi-Factor Logistic Regression achieves **95.68% accuracy** (+24.32% improvement), while Gradient Boosting achieves **97.12% accuracy** (+25.76% improvement) and an exceptional **0.9971 ROC-AUC**.

### 3.2. Probability Calibration & Brier Score
- The Brier score measures how accurately predicted probabilities correspond to true probabilities:
  - Baseline 2D Model: **0.1774** (poorly calibrated, severe probability distortion).
  - Multi-factor Logistic Regression: **0.0302** (highly calibrated).
  - Gradient Boosting: **0.0218** (outstanding calibration, sharp confidence).

### 3.3. Confusion Matrix Analysis (Gradient Boosting on 625 Holdout Samples)
```text
                    Predicted Declined    Predicted Approved
Actual Declined:           336                    13          (Specificity: 96.27%)
Actual Approved:             5                   271          (Sensitivity: 98.19%)
```
- **False Negative Approvals (Credit Risk / Type I Error):** Only 13 out of 349 declined applicants were misclassified as eligible by the pure ML model. In CrediEdge AI, our **Deterministic Rules Engine** catches these 13 edge cases through hard policy caps (e.g., CIBIL < 550 or DTI > 65%), eliminating false approvals!
- **False Declines (Revenue Opportunity Loss / Type II Error):** Only 5 eligible applicants were classified as declined, demonstrating a high 98.19% recall rate.

---

## 4. Prescriptive Adverse Action Verification

In addition to classification accuracy, the adverse action counterfactual optimization engine was verified across test decline cases:
1. **Income Optimization:** For an applicant earning ₹50,000 requesting ₹15,00,000 with a DTI of 144%, the engine accurately computed that a minimum income of ₹160,000 is required to service the requested debt at a safe 45% DTI.
2. **Loan Down-Sizing:** The engine accurately identified that the applicant could qualify for ₹3,70,000 at their current income.
3. **Tenure Extension:** For applicants requesting short 24-month loans, extending tenure to 60 or 84 months reduced the monthly EMI to successfully bring DTI within permissible limits.

---

## 5. Limitations & Future Work

1. **Macroeconomic Sensitivity:** The current dataset reflects stable interest rate cycles. Future iterations should incorporate stress-testing scenarios with rising benchmark rates.
2. **Alternative Data Sources:** Integration of utility payment history, bank statement open banking APIs (Account Aggregator framework), and GST data for self-employed profiles will further reduce credit assessment variance.
3. **Fairness & Bias Auditing:** Future releases will include automated disparate impact ratio calculations across demographic cohorts to guarantee compliance with fair lending mandates.
