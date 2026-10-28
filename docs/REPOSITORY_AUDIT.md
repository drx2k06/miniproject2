# Repository Audit & Forensic Analysis

**Target Project:** Mini-Project2  
**Source Repository:** [https://github.com/kirankumarnscr7-cmyk/Mini-Project2](https://github.com/kirankumarnscr7-cmyk/Mini-Project2)  
**Baseline Commit:** `1d304db7cad6e530af9e983660b1ee97d1d228f2`  
**Auditor:** Daniel Richard Xavier  
**Date:** October 28, 2026  
**Audited Directory:** `C:\Users\drx20\Projects\mini-project2-reference`  

---

## 1. Executive Summary

An exhaustive technical audit of the reference repository `Mini-Project2` was conducted. The project attempts to solve retail loan eligibility classification using Logistic Regression based on two applicant attributes: CIBIL credit score and requested loan amount.

While the fundamental domain problem (credit risk decisioning) is practical and relevant, the source implementation suffers from fatal execution bugs, extreme data sparsity, severe methodological flaws (100% data leakage, lack of validation sets), absence of real-world underwriting variables (e.g., Debt-to-Income, income, tenure, existing obligations), hardcoded procedural scripts with no modular separation, lack of testing, and zero documentation or licensing.

---

## 2. File-by-File Inventory & Forensic Inspection

The original repository contains exactly four files:

| File Path | Size (Bytes) | Lines | Purpose / Description | Status / Integrity |
| :--- | :--- | :--- | :--- | :--- |
| `eligibilityy.py` | 1,366 B | 29 | Logistic regression from scratch via `scipy.optimize.minimize` (BFGS) on synthetic random data. | **FATAL FAILURE** - Contains uncleaned Git diff header `@@ -0,0 +1,28 @@` on Line 1. Fails with `SyntaxError`. |
| `Li.py` | 2,021 B | 50 | Scikit-learn `LogisticRegression` pipeline on a 14-row hardcoded toy dataset. | **POOR QUALITY** - High overfitting risk, evaluates accuracy only on training data, blocks execution with `plt.show()`. |
| `Screenshot 2026-10-11 152725 (1).png` | 33,609 B | N/A | Terminal output screenshot showing sample predictions. | Raw unorganized binary artifact. |
| `Screenshot 2026-10-12 234117.png` | 119,385 B | N/A | Matplotlib decision boundary contour plot screenshot. | Raw unorganized binary artifact. |

---

## 3. Deep-Dive Code Analysis & Flaw Identification

### 3.1. Fatal Syntax Error in `eligibilityy.py`
Line 1 of `eligibilityy.py` starts with:
```python
@@ -0,0 +1,28 @@
import pandas as pd, numpy as np, matplotlib.pyplot as plt
```
When executed via `python eligibilityy.py` or compiled with `py_compile`, Python immediately halts with:
```text
SyntaxError: invalid syntax at line 1
```
This indicates the author copied a diff hunk directly from GitHub or git terminal into the file without testing or validating execution.

### 3.2. Extreme Data Sparsity & Methodological Flaws
- In `Li.py`, the dataset consists of only **14 rows**:
  - `Cibil`: `[520, 580, 610, 650, 650, 690, 720, 750, 780, 800, 560, 700, 740, 620]`
  - `LoanAmount`: `[5, 10, 8, 15, 5, 12, 25, 30, 20, 40, 18, 28, 22, 12]`
  - `Eligible`: `[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]`
- **100% Training Data Leakage:** The evaluation is computed via:
  ```python
  print("accuracy on training data:", round(accuracy_score(y, model.predict(x)), 2))
  ```
  No train/test split, stratified hold-out, or cross-validation exists. Evaluating a model exclusively on its training data violates core statistical learning principles and gives zero indication of out-of-sample generalization.
- In `eligibilityy.py`, the ground truth is generated via a deterministic step function:
  ```python
  df['eligible'] = ((df['cibil_score']>650) & (df['loan_amount']<1500000)).astype(int)
  ```
  Fitting an unregularized logistic sigmoid to perfectly separable step-function data leads to infinite weights (quasicomplete separation) and numerical instability.

### 3.3. Unrealistic Underwriting Modeling (Missing Key Financial Drivers)
In professional banking and retail lending (e.g., RBI, Fannie Mae, Basel III credit risk frameworks), loan eligibility is never determined by CIBIL and Loan Amount alone:
- **Missing Income & DTI:** Debt-to-Income (DTI) ratio is the single most critical factor in credit underwriting. A borrower with a 750 CIBIL score earning ₹25,000/month cannot service a ₹25,00,000 loan.
- **Missing Existing Obligations:** No feature accounts for existing monthly EMIs or outstanding liabilities.
- **Missing Tenure & Interest Rate:** Monthly installment burden directly depends on the loan repayment term (tenure in months) and interest rate.
- **Missing Employment & Loan Purpose:** Salaried vs. Self-Employed risk profiles and risk variations across Home, Personal, Education, and Auto loans are completely omitted.

### 3.4. Architectural & Production Deficiencies
- **Zero Modular Structure:** Everything is written as flat procedural scripts. No reusable classes, functions, or modules.
- **No Input Validation:** No bounds checking on input fields (negative loan amounts, CIBIL scores outside 300–900 range, zero tenure).
- **GUI / CLI Absence:** The only interface is modifying variables inside `.py` scripts and viewing GUI popups (`plt.show()`), which freeze automated test runs and headless environments.
- **No Model Serialization:** Trained weights and pipelines are discarded as soon as the Python process exits. No persistence (`joblib`, `onnx`, or `pickle`).
- **No Explainability or Counterfactual Guidance:** The scripts output a bare `0` or `1`. In real-world credit compliance, rejected applicants require Adverse Action notices detailing *why* they were rejected and *what specific actions* would make them eligible (e.g., reduce loan amount by ₹3 Lakhs, clear existing EMIs of ₹5,000, or extend tenure by 24 months).

---

## 4. Software Engineering & Governance Gaps

1. **No Dependency Specification:** No `requirements.txt`, `pyproject.toml`, or `Pipfile`.
2. **No Tests:** Zero unit tests, integration tests, or smoke tests.
3. **No License:** The repository has no `LICENSE` file. All rights are reserved by default; attribution and fresh independent implementation are strictly required.
4. **No Documentation:** No `README.md`, docstrings, type hints, or user manuals.
5. **No Versioning or Release Management:** Commit messages are generic auto-generated strings (`"Add files via upload"`).

---

## 5. Audit Recommendations & Modernization Strategy

| Original Weakness | Engineered Solution in CrediEdge AI |
| :--- | :--- |
| Broken syntax in `eligibilityy.py` (`@@` diff header) | Clean, robust Python 3.12 codebase with 100% compilation and automated linting. |
| 14-row toy data / synthetic deterministic rule | Multi-dimensional credit risk engine with realistic financial parameters (CIBIL, Income, Loan Amount, Tenure, Existing EMIs, DTI, Employment). |
| 100% data leakage (train-only evaluation) | Stratified 80/20 train/test splits, 5-fold cross-validation, ROC-AUC, Brier score, Precision/Recall, and Confusion Matrix. |
| Single uncalibrated model | Multi-model benchmark: Custom Scratch Logistic Regression (regularized BFGS/SGD), Scikit-Learn Calibrated Pipeline, and Random Forest Ensemble. |
| Procedural spaghetti scripts | Clean 3-tier architecture: Domain Layer (`src/engine`), ML & Pipeline Layer (`src/models`), Presentation Layer (`src/ui` + CLI). |
| Black-box binary decision (`0` / `1`) | 3-Tier Risk Classification (`Approved`, `Manual Review`, `Declined`) + Counterfactual Adverse Action Guidance for rejected applicants. |
| Blocking `plt.show()` popups | High-performance Streamlit Web Application + Command-Line Interface (`src/cli.py`) with interactive SVG/HTML charts. |
| Zero test coverage | Automated `pytest` suite with unit, regression, domain validation, and edge-case tests. |
