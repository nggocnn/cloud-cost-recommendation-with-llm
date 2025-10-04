"""
Simple CLI functionality tests.
"""

import pytest
import subprocess


@pytest.mark.functional
@pytest.mark.cli
def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        ["python", "-m", "llm_cost_recommendation", "--help"],
        capture_output=True,
        text=True,
        cwd="/home/nggocnn/llm-cost-recommendation",
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "llm cost optimization" in result.stdout.lower()


@pytest.mark.functional
@pytest.mark.cli
def test_cli_analyze_help():
    """Test CLI analyze subcommand help."""
    result = subprocess.run(
        ["python", "-m", "llm_cost_recommendation", "analyze", "--help"],
        capture_output=True,
        text=True,
        cwd="/home/nggocnn/llm-cost-recommendation",
    )

    assert result.returncode == 0
    assert "sample-data" in result.stdout
    assert "billing-file" in result.stdout
