"""Integration tests for the CrediEdge AI Click CLI."""
from click.testing import CliRunner
import pytest
from src.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    """Verify that CLI help displays all registered commands."""
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "generate-data" in res.output
    assert "train" in res.output
    assert "evaluate" in res.output
    assert "predict" in res.output
    assert "run-all" in res.output


def test_cli_generate_data(runner, tmp_path):
    """Verify data generation command creates a valid CSV file."""
    out_file = tmp_path / "test_data.csv"
    res = runner.invoke(main, ["generate-data", "--n-samples", "150", "--output", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()


def test_cli_predict_approved(runner):
    """Verify predict command outputs formatted report for approved applicant."""
    res = runner.invoke(main, [
        "predict",
        "--cibil", "790",
        "--income", "120000",
        "--amount", "400000",
        "--tenure", "36",
        "--emis", "5000",
    ])
    assert res.exit_code == 0
    assert "UNDERWRITING DECISION REPORT" in res.output
    assert "APPROVED" in res.output
    assert "LOW RISK" in res.output.upper()


def test_cli_predict_declined_with_adverse_action(runner):
    """Verify predict command outputs adverse action guidance for declined applicant."""
    res = runner.invoke(main, [
        "predict",
        "--cibil", "520",  # Subprime
        "--income", "40000",
        "--amount", "1000000",
        "--tenure", "24",
        "--emis", "15000",
    ])
    assert res.exit_code == 0
    assert "UNDERWRITING DECISION REPORT" in res.output
    assert "DECLINED" in res.output
    assert "ACTIONABLE ADVERSE ACTION GUIDANCE" in res.output
