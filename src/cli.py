"""
CrediEdge AI Command-Line Interface (CLI).
"""

import sys
import subprocess
from pathlib import Path
import click
import pandas as pd
import json

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator import generate_synthetic_credit_dataset
from src.data.schema import ApplicantProfile, EmploymentType, LoanPurpose
from src.engine.rules_engine import UnderwritingRulesEngine
from src.models.pipeline import CrediEdgePipeline, CrediEdgePredictor
from src.models.evaluation import compare_models


@click.group()
def main():
    """CrediEdge AI: Autonomous Credit Risk Underwriting & Loan Decisioning Engine."""
    pass


@main.command(name="generate-data")
@click.option("--n-samples", default=3000, help="Number of synthetic credit applicant records.")
@click.option("--output", default="data/credit_dataset.csv", help="Output CSV path.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def generate_data(n_samples, output, seed):
    """Generates realistic retail credit dataset reflecting commercial banking distributions."""
    click.echo(f"[*] Generating {n_samples} credit underwriting records (seed={seed})...")
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_credit_dataset(n_samples=n_samples, random_seed=seed, output_path=str(out_path))
    click.echo(f"[+] Dataset successfully saved to: {out_path}")
    click.echo(f"    Shape: {df.shape} | Eligible: {df['eligible'].sum()} ({(df['eligible'].mean()*100):.1f}%)")


@main.command(name="train")
@click.option("--data-path", default="data/credit_dataset.csv", help="Path to credit dataset CSV.")
@click.option("--artifacts-dir", default="artifacts", help="Directory to save model artifacts.")
def train(data_path, artifacts_dir):
    """Trains full suite of underwriting models and saves artifacts."""
    d_path = PROJECT_ROOT / data_path
    if not d_path.exists():
        click.echo(f"[*] Data file not found. Generating default dataset at {d_path}...")
        d_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_credit_dataset(n_samples=2500, random_seed=42, output_path=str(d_path))

    click.echo(f"[*] Loading training dataset from: {d_path}")
    df = pd.read_csv(d_path)

    art_dir = PROJECT_ROOT / artifacts_dir
    click.echo(f"[*] Fitting benchmark model suite and serializing to: {art_dir}")
    results_df, metrics_dict, models = compare_models(
        df=df, test_size=0.25, random_state=42, artifacts_dir=art_dir
    )

    click.echo("\n[+] Model Suite Training Complete! Holdout Evaluation:")
    for _, row in results_df.iterrows():
        click.echo(f"  * {row['model_name']:25}: Accuracy={row['accuracy']*100:.2f}% | ROC-AUC={row['roc_auc']:.4f} | Brier={row['brier_score']:.4f}")


@main.command(name="evaluate")
@click.option("--data-path", default="data/credit_dataset.csv", help="Path to credit dataset CSV.")
def evaluate(data_path):
    """Evaluates all underwriting models and prints comprehensive metrics."""
    d_path = PROJECT_ROOT / data_path
    if not d_path.exists():
        click.echo("[*] Generating dataset for evaluation...")
        d_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_credit_dataset(n_samples=2000, random_seed=42, output_path=str(d_path))

    df = pd.read_csv(d_path)
    results_df, _, _ = compare_models(df=df, test_size=0.25, random_state=42)

    click.echo("\n" + "=" * 80)
    click.echo("CREDIEDGE AI — COMPREHENSIVE UNDERWRITING MODEL BENCHMARK")
    click.echo("=" * 80)
    cols = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "brier_score"]
    click.echo(results_df[cols].to_string(index=False))
    click.echo("=" * 80)


@main.command(name="predict")
@click.option("--cibil", default=740, help="CIBIL Credit Score (300-900).")
@click.option("--income", default=75000.0, help="Monthly Net Income (INR).")
@click.option("--amount", default=500000.0, help="Requested Loan Principal (INR).")
@click.option("--tenure", default=36, help="Repayment Duration in Months.")
@click.option("--emis", default=10000.0, help="Existing Monthly Debt EMIs (INR).")
@click.option("--rate", default=11.5, help="Annual Interest Rate (%).")
@click.option("--employment", default="Salaried", help="Employment Type.")
@click.option("--purpose", default="Personal", help="Loan Purpose.")
def predict(cibil, income, amount, tenure, emis, rate, employment, purpose):
    """Evaluates an applicant profile through regulatory rules, ML scoring, and counterfactuals."""
    art_path = PROJECT_ROOT / "artifacts" / "gradient_boosting.joblib"
    if not art_path.exists():
        click.echo("[*] Artifacts not found. Training model suite first...")
        ctx = click.get_current_context()
        ctx.invoke(train)

    pipeline = CrediEdgePipeline.load_artifact(art_path)
    predictor = CrediEdgePredictor(model_pipeline=pipeline)

    app = ApplicantProfile(
        cibil_score=cibil,
        monthly_income=income,
        loan_amount=amount,
        loan_tenure_months=tenure,
        existing_monthly_emis=emis,
        interest_rate_annual=rate,
        employment_type=EmploymentType(employment),
        loan_purpose=LoanPurpose(purpose),
    )

    decision = predictor.predict(app)

    click.echo("\n" + "=" * 60)
    click.echo("CREDIEDGE AI — UNDERWRITING DECISION REPORT")
    click.echo("=" * 60)
    click.echo(f"  * Status               : {decision.status.value.upper()}")
    click.echo(f"  * Risk Tier            : {decision.risk_tier.value}")
    click.echo(f"  * Approval Probability : {decision.approval_probability * 100:.2f}%")
    click.echo(f"  * Proposed Monthly EMI : INR {decision.metrics.proposed_emi:,.2f}")
    click.echo(f"  * Debt-To-Income (DTI) : {decision.metrics.dti_ratio * 100:.2f}%")
    click.echo(f"  * FOIR                 : {decision.metrics.foir * 100:.2f}%")

    if decision.hard_stops_triggered:
        click.echo("\n  [!] HARD STOPS TRIGGERED:")
        for hs in decision.hard_stops_triggered:
            click.echo(f"      - {hs}")

    if decision.counterfactual_advice:
        cf = decision.counterfactual_advice
        click.echo("\n  [+] ACTIONABLE ADVERSE ACTION GUIDANCE:")
        click.echo(f"      - Max Eligible Loan Amount : INR {cf.max_permissible_loan:,.2f}")
        click.echo(f"      - Min Income Required      : INR {cf.min_required_income:,.2f}")
        click.echo(f"      - Recommended Tenure Ext   : {cf.required_tenure_extension_months} months")
        click.echo(f"      - Existing Debt Paydown    : INR {cf.required_debt_paydown:,.2f}")
    click.echo("=" * 60)


@main.command(name="dashboard")
@click.option("--port", default=8502, help="Port to run Streamlit on.")
def dashboard(port):
    """Starts the interactive Streamlit Underwriting Dashboard."""
    ui_script = PROJECT_ROOT / "src" / "ui" / "app.py"
    click.echo(f"[*] Starting CrediEdge AI Dashboard on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_script), "--server.port", str(port)])


@main.command(name="run-all")
def run_all():
    """Executes the complete pipeline: generation, training, evaluation, and sample prediction."""
    click.echo("[*] Starting Complete CrediEdge AI Workflow...")
    ctx = click.get_current_context()
    ctx.invoke(generate_data, n_samples=2500)
    ctx.invoke(train)
    ctx.invoke(evaluate)
    ctx.invoke(predict, cibil=750, income=85000, amount=600000, tenure=36, emis=10000)
    click.echo("\n[SUCCESS] End-to-End CrediEdge AI Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
