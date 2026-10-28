# Improvement Roadmap & Engineering Evolution

**Project:** CrediEdge AI (Enterprise Loan Eligibility Decisioning)  
**Lead Developer:** Daniel Richard Xavier  
**Version:** 2.0.0  

---

## 1. Motivation & Strategic Objectives

The reference repository (`kirankumarnscr7-cmyk/Mini-Project2`) contained basic experimental scripts attempting binary classification on toy datasets. The goal of this re-engineering initiative is to transform those rough concepts into a mathematically sound, production-grade financial platform.

---

## 2. Multi-Phase Roadmap & Delivery Matrix

| Phase | Focus Area | Source Limitation | CrediEdge AI Modernization | Status |
| :---: | :--- | :--- | :--- | :---: |
| **P1** | **Repository Safety & Audit** | No documentation, unknown state, fatal syntax error in `eligibilityy.py`. | Safe isolation clone, comprehensive audit (`REPOSITORY_AUDIT.md`), git initialization. | **COMPLETE** |
| **P2** | **Architecture & Design** | Procedural scripts, zero modularity, hardcoded constants. | 3-tier modular architecture (`ARCHITECTURE.md`), separation of domain logic, ML, data, and UI. | **COMPLETE** |
| **P3** | **Data Modeling & Domain Engine** | Only 2 features (CIBIL, Loan Amount), 14 rows, no realistic banking metrics. | Realistic financial schemas (`schema.py`), multi-factor risk data generator (DTI, Income, Tenure, EMIs, Employment). | **COMPLETE** |
| **P4** | **Financial Math & Rules Engine** | Binary 0/1 output, no banking compliance or policy checks. | Amortization schedules, DTI/FOIR calculators, 3-tier risk decisioning (`Approved`, `Manual Review`, `Declined`). | **COMPLETE** |
| **P5** | **Adverse Action & Explainability** | Zero feedback for rejected applicants. | Counterfactual optimization engine generating actionable advice to qualify for credit. | **COMPLETE** |
| **P6** | **ML Engine & Rigorous Modeling** | Broken BFGS / 100% training data leakage. | Custom regularized Logistic Regression from scratch + Scikit-Learn Calibrated Pipeline + Random Forest benchmark. | **COMPLETE** |
| **P7** | **Model Evaluation & Reporting** | Single training accuracy number. | Stratified 5-fold CV, ROC-AUC, Brier score, Confusion Matrix, `EVALUATION_REPORT.md`. | **COMPLETE** |
| **P8** | **Interactive Web Application** | Blocking `plt.show()` popups. | Full-stack interactive Streamlit dashboard with custom financial styling, gauges, amortization charts, and batch tools. | **COMPLETE** |
| **P9** | **Automated Test Suite** | 0% test coverage. | Unit and regression test suite using `pytest` covering financial math, ML estimators, edge cases, and CLI (31 tests passing). | **COMPLETE** |
| **P10** | **Documentation & Publication** | No README or licensing. | Comprehensive enterprise README, clear attribution, and GitHub readiness. | **COMPLETE** |

---

## 3. High-Impact Technical Decisions

1. **Why multi-factor underwriting instead of just CIBIL and Loan Amount?**
   In retail credit underwriting, approval is fundamentally about debt serviceability. A borrower with high credit score but insufficient disposable income or excessive existing leverage is a severe default risk. Adding monthly income, tenure, existing obligations, and DTI creates an authentic financial system.

2. **Why both a custom scratch estimator and production Scikit-Learn pipelines?**
   The source project attempted two scripts: one from scratch (`eligibilityy.py`) and one using Scikit-Learn (`Li.py`). To honor and elevate both concepts, we implement:
   - A fully verified, numerically stable mathematical implementation of Logistic Regression from scratch (with L2 regularization and BFGS/SGD solvers).
   - An industrial Scikit-Learn pipeline supporting categorical encoding, feature scaling, probability calibration, and ensemble comparisons.

3. **Why Adverse Action / Counterfactuals?**
   Under consumer credit regulations (such as Equal Credit Opportunity Act, Fair Credit Reporting Act, and RBI customer protection directions), financial institutions must provide applicants with specific reasons for credit denial. Our counterfactual engine solves the inverse optimization problem to tell applicants exactly how to qualify.
