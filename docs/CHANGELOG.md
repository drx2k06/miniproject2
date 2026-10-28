# Changelog & Engineering Evolution

All notable changes to the **CrediEdge AI** platform are documented in this file.

---

## [2.0.0] — 2026-10-28
### Re-engineered and Published by Daniel Richard Xavier

#### 🚀 Added
- **Domain & Financial Underwriting Engine (`src/engine/`):**
  - `financial_math.py`: Accurate commercial banking reducing-balance EMI calculator, Debt-to-Income (DTI) ratio, Fixed Obligation to Income Ratio (FOIR), maximum eligible loan inversion, and complete monthly amortization schedules.
  - `rules_engine.py`: Deterministic underwriting policy engine enforcing regulatory guardrails, CIBIL cutoffs (prime vs subprime), hard DTI caps (65%), and loan-to-income (LTI) restrictions.
  - `adverse_action.py`: Automated counterfactual explainability engine providing actionable adverse action recommendations for rejected borrowers (maximum qualifying loan amount, required income, tenure extensions, and debt paydown).
- **Data & Schema Layer (`src/data/`):**
  - `schema.py`: Type-safe Pydantic v2 schemas for applicant profiles, financial metrics, risk tiers, decision statuses, and counterfactual advice.
  - `generator.py`: Realistic synthetic credit dataset generator reflecting commercial banking distributions (income log-normal, bimodal CIBIL scores, DTI distributions, stochastic noise, and policy vetoes).
- **Machine Learning & Modeling (`src/models/`):**
  - `scratch_logistic.py`: Clean, numerically stable mathematical implementation of Logistic Regression from scratch, featuring L2 regularization (Ridge), exponent-clipped sigmoid, and dual solvers (Scipy BFGS and Gradient Descent).
  - `pipeline.py`: Production Scikit-Learn pipelines combining `StandardScaler`, `OneHotEncoder`, class-balanced Logistic Regression, Random Forest, and Gradient Boosting.
  - `evaluation.py`: Comprehensive out-of-sample evaluation suite computing Accuracy, Precision, Recall, F1-Score, ROC-AUC, Brier score, and Confusion Matrices.
- **Presentation & Interface Layer:**
  - `src/ui/app.py`: Modern, high-performance interactive Streamlit dashboard featuring Single Applicant Underwriting, Decision Boundary 2D visualization, Model Benchmarks, and Batch Underwriting.
  - `src/cli.py`: Unified Click-based CLI with commands: `generate-data`, `train`, `evaluate`, `predict`, `dashboard`, and `run-all`.
- **Testing & Quality Assurance (`tests/`):**
  - Complete automated test suite with 31 unit, integration, and CLI tests passing with 100% success rate across financial math, policy rules, adverse action, data generation, scratch ML, pipelines, and CLI.
- **Documentation & Packaging:**
  - Comprehensive `README.md`, `docs/REPOSITORY_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/IMPROVEMENT_ROADMAP.md`, `docs/EVALUATION_REPORT.md`, `docs/FINAL_REPORT.md`.
  - `pyproject.toml`, `setup.py`, `requirements.txt`, `.gitignore`, `.env.example`, and MIT `LICENSE`.

#### 🐛 Fixed
- Fixed fatal `SyntaxError` on Line 1 of reference `eligibilityy.py` (`@@ -0,0 +1,28 @@`).
- Fixed 100% training data leakage by implementing stratified train/test holdouts.
- Fixed unregularized logistic optimization failures and numerical overflow in BFGS cost function.
- Eliminated blocking GUI popups (`plt.show()`) that stalled automation in headless environments.

---

## [1.0.0] — 2026-10-15
### Original Reference Prototype by Kiran Kumar N (`kirankumarnscr7-cmyk`)

#### Baseline Implementation
- `eligibilityy.py`: Scratch logistic regression script using `scipy.optimize.minimize` on 300 synthetic points with deterministic rectangular rule.
- `Li.py`: Scikit-learn Logistic Regression script fitted and evaluated on 14 toy rows with 2 features (`Cibil`, `LoanAmount`).
- Static screenshots of terminal output and matplotlib contour plot.
