import json
import subprocess
import sys

import pytest

from transport_audit.experiment import run_audit


def test_audit_is_deterministic_and_checks_all_solves():
    summary, plans = run_audit(9)
    again, again_plans = run_audit(9)
    assert summary == again
    assert plans == again_plans
    assert len(summary["results"]) == 12
    assert summary["checks"]["number_of_solves"] == 38
    assert summary["checks"]["max_rounded_marginal_error"] < 1e-13
    assert summary["checks"]["max_primal_dual_gap"] <= 1e-9
    assert summary["offset_stress"]["naive_kernel_zero_fraction"] == 1.0


@pytest.mark.parametrize("bins", [True, 1, 101, 9.5])
def test_invalid_experiment_size(bins):
    with pytest.raises(ValueError):
        run_audit(bins)


def test_cli_writes_usable_json_and_pngs(tmp_path):
    subprocess.run([sys.executable, "-m", "transport_audit.experiment", "--bins", "9",
                    "--output", str(tmp_path)], check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    plans = json.loads((tmp_path / "couplings.json").read_text())
    assert summary["configuration"]["bins"] == 9
    assert len(plans["lp"]) == 9
    for name in ("self-cost-correction.png", "transport-plans.png"):
        content = (tmp_path / name).read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(content) > 10000
