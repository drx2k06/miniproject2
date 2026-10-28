"""
CrediEdge AI: Interactive Credit Risk Underwriting & Loan Eligibility Platform.
Developer: Daniel Richard Xavier
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import (
    ApplicantProfile,
    EmploymentType,
    LoanPurpose,
    DecisionStatus,
    RiskTier,
)
from src.data.generator import generate_synthetic_credit_dataset
from src.engine.rules_engine import UnderwritingRulesEngine
from src.engine.financial_math import calculate_monthly_emi, generate_amortization_schedule
from src.models.pipeline import CrediEdgePipeline, CrediEdgePredictor
from src.models.evaluation import compare_models
from src.models.scratch_logistic import ScratchLogisticRegression

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CrediEdge AI | Credit Risk & Loan Eligibility Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End FinTech Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    .fintech-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-approved {
        background-color: #065f46;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 18px;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-review {
        background-color: #78350f;
        color: #fbbf24;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 18px;
        display: inline-block;
        border: 1px solid #d97706;
    }
    .badge-declined {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 18px;
        display: inline-block;
        border: 1px solid #dc2626;
    }
    .counterfactual-box {
        background-color: #172554;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_system():
    """Initializes dataset and trains benchmark model suite."""
    df = generate_synthetic_credit_dataset(n_samples=2500, random_seed=42)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_df, metrics_dict, trained_models = compare_models(
        df=df, test_size=0.25, random_state=42, artifacts_dir=artifacts_dir
    )
    # Production predictor using best gradient boosting model
    rules_engine = UnderwritingRulesEngine()
    predictor = CrediEdgePredictor(
        model_pipeline=trained_models["gradient_boosting"],
        rules_engine=rules_engine,
    )
    return df, results_df, trained_models, predictor


df_dataset, results_df, trained_models, predictor = load_trained_system()

# Application Header
st.markdown('<span style="background-color:#0284c7; color:white; padding:4px 10px; border-radius:4px; font-weight:bold; font-size:12px;">ENTERPRISE UNDERWRITING ENGINE</span>', unsafe_allow_html=True)
st.title("💳 CrediEdge AI — Autonomous Credit Risk & Loan Eligibility Decisioning")
st.caption("Engineered by **Daniel Richard Xavier** | Original Research Reference: Kiran Kumar N (`kirankumarnscr7-cmyk`)")

# Sidebar Controls
st.sidebar.image("https://img.shields.io/badge/CrediEdge_AI-v2.0.0-0284c7?style=for-the-badge", use_container_width=True)
st.sidebar.markdown("### ⚙️ Underwriting Policy Configuration")
policy_min_cibil = st.sidebar.slider("Policy Minimum CIBIL Cutoff", 300, 700, 550, 10)
policy_max_dti = st.sidebar.slider("Policy Hard-Stop DTI Ratio (%)", 40, 80, 65, 5) / 100.0
st.sidebar.markdown("---")
st.sidebar.info("CrediEdge AI integrates deterministic regulatory vetoes with calibrated machine learning probability scoring and prescriptive counterfactual explainability.")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📋 Single Applicant Underwriting",
    "📈 Decision Boundary & Model Benchmarks",
    "📂 Portfolio & Batch Underwriting",
])

# ==================== TAB 1: Single Applicant Underwriting ====================
with tab1:
    st.subheader("Interactive Loan Application & Underwriting Evaluation")

    c1, c2, c3 = st.columns(3)
    with c1:
        app_id = st.text_input("Applicant Identifier", value="APP-8942")
        cibil = st.slider("CIBIL Credit Score", 300, 900, 720, 5)
        age = st.number_input("Applicant Age (Years)", min_value=18, max_value=75, value=34)
        emp_type = st.selectbox("Employment Type", [e.value for e in EmploymentType])
    with c2:
        monthly_inc = st.number_input("Monthly Net Income (₹)", min_value=15000.0, max_value=2000000.0, value=75000.0, step=5000.0)
        loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=25000.0, max_value=20000000.0, value=800000.0, step=25000.0)
        tenure = st.selectbox("Repayment Duration (Months)", [12, 24, 36, 48, 60, 84, 120, 180, 240, 300], index=3)
        purpose = st.selectbox("Loan Purpose", [p.value for p in LoanPurpose])
    with c3:
        existing_emis = st.number_input("Existing Monthly Debt EMIs (₹)", min_value=0.0, max_value=500000.0, value=12000.0, step=1000.0)
        int_rate = st.number_input("Annual Interest Rate (%)", min_value=5.0, max_value=28.0, value=10.5, step=0.25)
        st.markdown("<br>", unsafe_allow_html=True)
        eval_btn = st.button("🚀 Evaluate Underwriting Decision", use_container_width=True, type="primary")

    applicant = ApplicantProfile(
        applicant_id=app_id,
        age=age,
        cibil_score=cibil,
        monthly_income=monthly_inc,
        loan_amount=loan_amt,
        loan_tenure_months=tenure,
        existing_monthly_emis=existing_emis,
        interest_rate_annual=int_rate,
        employment_type=EmploymentType(emp_type),
        loan_purpose=LoanPurpose(purpose),
    )

    decision = predictor.predict(applicant)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        st.markdown("### 🏆 Decision & Risk Classification")
        if decision.status == DecisionStatus.APPROVED:
            st.markdown(f'<div class="badge-approved">✅ LOAN DECISION: {decision.status.value.upper()}</div>', unsafe_allow_html=True)
        elif decision.status == DecisionStatus.MANUAL_REVIEW:
            st.markdown(f'<div class="badge-review">⚠️ LOAN DECISION: {decision.status.value.upper()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-declined">❌ LOAN DECISION: {decision.status.value.upper()}</div>', unsafe_allow_html=True)

        st.markdown(f"**Risk Assignment:** `{decision.risk_tier.value}`")
        st.markdown(f"**Approval Probability:** `{decision.approval_probability * 100:.2f}%`")
        st.progress(decision.approval_probability)

        # Policy checks summary
        st.markdown("#### 🛡️ Underwriting Governance Checks")
        if not decision.hard_stops_triggered and not decision.policy_warnings:
            st.success("✔️ All regulatory and risk governance checks satisfied.")
        else:
            for hs in decision.hard_stops_triggered:
                st.error(f"🚫 **HARD STOP:** {hs}")
            for pw in decision.policy_warnings:
                st.warning(f"⚠️ **WARNING:** {pw}")

    with res_col2:
        st.markdown("### 📊 Debt Serviceability & Key Ratios")
        m = decision.metrics
        met_c1, met_c2 = st.columns(2)
        with met_c1:
            st.markdown(
                f"""<div class="fintech-card"><div class="metric-label">Proposed Monthly EMI</div>
                <div class="metric-value">₹{m.proposed_emi:,.2f}</div>
                <small>Total Debt Servicing: ₹{m.total_monthly_debt:,.2f}</small></div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="fintech-card"><div class="metric-label">Debt-To-Income (DTI)</div>
                <div class="metric-value">{m.dti_ratio * 100:.1f}%</div>
                <small>Target: ≤ 45% | Hard Cap: 65%</small></div>""",
                unsafe_allow_html=True,
            )
        with met_c2:
            st.markdown(
                f"""<div class="fintech-card"><div class="metric-label">Total Interest Payable</div>
                <div class="metric-value">₹{m.total_interest_payable:,.2f}</div>
                <small>Total Repayment: ₹{m.total_payment:,.2f}</small></div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="fintech-card"><div class="metric-label">Loan-To-Income Multiple</div>
                <div class="metric-value">{m.loan_to_income_multiple:.1f}x</div>
                <small>Annual Income: ₹{m.annual_income:,.2f}</small></div>""",
                unsafe_allow_html=True,
            )

    # Adverse action counterfactual advice if declined or review
    if decision.counterfactual_advice:
        cf = decision.counterfactual_advice
        st.markdown("### 💡 Prescriptive Adverse Action Guidance (How to Qualify)")
        st.markdown(
            f"""
            <div class="counterfactual-box">
                <b>Regulatory Adverse Action Notice & Remedial Pathway:</b>
                <ul>
                    <li><b>Maximum Permissible Loan at Current Profile:</b> ₹{cf.max_permissible_loan:,.2f} (Requested ₹{applicant.loan_amount:,.2f})</li>
                    <li><b>Required Minimum Monthly Income to Service Current Loan:</b> ₹{cf.min_required_income:,.2f} (Current ₹{applicant.monthly_income:,.2f})</li>
                    <li><b>Recommended Tenure Extension:</b> Extend to <b>{cf.required_tenure_extension_months} months</b> to bring DTI under acceptable limits.</li>
                    <li><b>Required Debt Paydown:</b> Reduce existing monthly debt EMIs by <b>₹{cf.required_debt_paydown:,.2f}</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📄 View Detailed Monthly Amortization Schedule (First 12 Months)"):
        sched = generate_amortization_schedule(applicant.loan_amount, applicant.interest_rate_annual, applicant.loan_tenure_months)
        st.dataframe(pd.DataFrame(sched[:12]), use_container_width=True)

# ==================== TAB 2: Decision Boundary & Model Benchmarks ====================
with tab2:
    st.subheader("Model Benchmarks & Mathematical Decision Boundaries")
    st.write(
        "Demonstrating the technical evolution from the original 2-feature toy prototype "
        "to industrial multi-factor underwriting engines."
    )

    st.markdown("#### 🏆 Out-of-Sample Performance Comparison (Holdout Test Set)")
    display_cols = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "brier_score"]
    styled_results = results_df[display_cols].copy()
    styled_results["accuracy"] = (styled_results["accuracy"] * 100).round(2).astype(str) + "%"
    styled_results["roc_auc"] = styled_results["roc_auc"].round(4)
    st.table(styled_results)

    # Decision Boundary 2D Visualization
    st.markdown("#### 🔬 Decision Boundary Visualization (CIBIL vs Loan Amount)")
    st.caption("Fitted on 2D space comparing Scratch Logistic Regression (BFGS) against ground truth eligibility.")

    scratch_model = trained_models["scratch_baseline_2d"].named_steps["classifier"]
    scaler = trained_models["scratch_baseline_2d"].named_steps["scaler"]

    # Generate synthetic 2D grid for decision boundary
    x_cibil = np.linspace(300, 900, 200)
    y_loan = np.linspace(50000, 3000000, 200)
    xx, yy = np.meshgrid(x_cibil, y_loan)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_scaled = scaler.transform(grid_points)
    probs = scratch_model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    contour = ax.contourf(xx, yy / 100000, probs, levels=20, cmap="RdYlGn", alpha=0.3)
    ax.contour(xx, yy / 100000, probs, levels=[0.5], colors="blue", linestyles="--", linewidths=2)

    # Plot sample data points
    sample_sub = df_dataset.sample(300, random_state=42)
    ax.scatter(
        sample_sub.loc[sample_sub["eligible"] == 1, "cibil_score"],
        sample_sub.loc[sample_sub["eligible"] == 1, "loan_amount"] / 100000,
        c="#22c55e", label="Eligible (Approved)", alpha=0.6, edgecolors="none", s=25
    )
    ax.scatter(
        sample_sub.loc[sample_sub["eligible"] == 0, "cibil_score"],
        sample_sub.loc[sample_sub["eligible"] == 0, "loan_amount"] / 100000,
        c="#ef4444", label="Not Eligible (Declined)", alpha=0.6, edgecolors="none", s=25
    )
    ax.set_xlabel("CIBIL Credit Score")
    ax.set_ylabel("Loan Amount (Lakhs INR)")
    ax.set_title("Scratch Logistic Regression Decision Boundary & Probability Surface")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for text in ax.legend().get_texts():
        text.set_color("#e2e8f0")
    st.pyplot(fig)

# ==================== TAB 3: Portfolio & Batch Underwriting ====================
with tab3:
    st.subheader("Batch Portfolio Underwriting & Risk Exposure Analytics")

    batch_size = st.slider("Select Batch Size of Applicants to Underwrite", 50, 1000, 200, 50)
    batch_df = df_dataset.sample(batch_size, random_state=123).copy().reset_index(drop=True)

    # Run batch evaluation through predictor
    batch_decisions = []
    for _, row in batch_df.iterrows():
        app = ApplicantProfile(
            cibil_score=int(row["cibil_score"]),
            monthly_income=float(row["monthly_income"]),
            loan_amount=float(row["loan_amount"]),
            loan_tenure_months=int(row["loan_tenure_months"]),
            existing_monthly_emis=float(row["existing_monthly_emis"]),
            interest_rate_annual=float(row["interest_rate_annual"]),
            employment_type=EmploymentType(row["employment_type"]),
            loan_purpose=LoanPurpose(row["loan_purpose"]),
        )
        dec = predictor.predict(app)
        batch_decisions.append({
            "CIBIL": app.cibil_score,
            "Income": app.monthly_income,
            "LoanAmount": app.loan_amount,
            "EMI": dec.metrics.proposed_emi,
            "DTI": round(dec.metrics.dti_ratio * 100, 1),
            "Status": dec.status.value,
            "RiskTier": dec.risk_tier.value,
            "ApprovalProb": round(dec.approval_probability, 3),
        })

    b_res = pd.DataFrame(batch_decisions)

    # Portfolio metrics
    n_approved = (b_res["Status"] == "Approved").sum()
    n_review = (b_res["Status"] == "Manual Review").sum()
    n_declined = (b_res["Status"] == "Declined").sum()
    app_rate = (n_approved / len(b_res)) * 100.0

    bp1, bp2, bp3, bp4 = st.columns(4)
    with bp1:
        st.metric("Total Batch Applications", len(b_res))
    with bp2:
        st.metric("Approval Rate", f"{app_rate:.1f}%")
    with bp3:
        st.metric("Manual Review Rate", f"{(n_review/len(b_res))*100:.1f}%")
    with bp4:
        st.metric("Total Requested Volume", f"₹{b_res['LoanAmount'].sum():,.0f}")

    st.markdown("#### Portfolio Applications Summary Table")
    st.dataframe(b_res.head(20), use_container_width=True)

    # Download CSV
    csv_bytes = b_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Complete Underwriting Decisions CSV",
        data=csv_bytes,
        file_name="crediedge_underwriting_decisions.csv",
        mime="text/csv",
    )
