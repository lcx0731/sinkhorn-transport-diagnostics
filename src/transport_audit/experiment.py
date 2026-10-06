"""Deterministic finite-histogram audit; no external data or fitted parameters."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .divergence import sinkhorn_divergence
from .solver import sinkhorn, squared_cost, unregularized_ot

CASES = (("identical", 0.0, 0.35), ("location_shift", 0.4, 0.35),
         ("location_and_scale", 0.4, 0.65))
EPSILONS = (0.03, 0.1, 0.3, 1.0)


def histogram(grid, center, width):
    mass = np.exp(-0.5 * ((grid - center) / width) ** 2)
    return mass / mass.sum()


def diagnostics(result):
    return {name: getattr(result, name) for name in (
        "iterations", "raw_marginal_error", "rounded_marginal_error",
        "transport_cost", "kl", "primal", "dual", "gap")}


def run_audit(bins=25):
    if isinstance(bins, bool) or not isinstance(bins, int) or not 5 <= bins <= 100:
        raise ValueError("bins must be an integer from 5 to 100")
    grid = np.linspace(-1.5, 1.5, bins)
    a = histogram(grid, 0, 0.35)
    cost = squared_cost(grid, grid)
    rows, plans = [], {}
    all_results = []
    for case, center, width in CASES:
        b = histogram(grid, center, width)
        lp = unregularized_ot(a, b, cost)
        for epsilon in EPSILONS:
            divergence = sinkhorn_divergence(grid, a, grid, b, epsilon)
            all_results.extend((divergence.cross, divergence.self_a, divergence.self_b))
            if divergence.cross.transport_cost < lp.cost - 1e-8:
                raise ArithmeticError("regularized plan beats the unregularized LP beyond tolerance")
            if divergence.lower > divergence.upper + 1e-10:
                raise ArithmeticError("inconsistent numerical divergence bracket")
            if divergence.lower < -1e-8:
                raise ArithmeticError("unexpected negative squared-cost divergence")
            rows.append({"case": case, "target_center": center, "target_width": width,
                         "epsilon": epsilon, "lp_cost": lp.cost,
                         "lp_marginal_error": lp.marginal_error,
                         "cross": diagnostics(divergence.cross),
                         "self_a": diagnostics(divergence.self_a),
                         "self_b": diagnostics(divergence.self_b),
                         "divergence": divergence.value,
                         "divergence_lower": divergence.lower,
                         "divergence_upper": divergence.upper})
            if case == "location_and_scale" and epsilon in (0.03, 1.0):
                plans[str(epsilon)] = divergence.cross.plan.tolist()
        if case == "location_and_scale":
            plans["lp"] = lp.plan.tolist()
            plans["target"] = b.tolist()
    # Adding a constant shifts every feasible objective by that constant.
    # A naive uncentered exp(-C/epsilon) now underflows completely.
    target = histogram(grid, 0.4, 0.65)
    offset, epsilon = 1000.0, 0.1
    base = sinkhorn(a, target, cost, epsilon)
    shifted = sinkhorn(a, target, cost + offset, epsilon)
    with np.errstate(under="ignore"):
        naive_kernel = np.exp(-(cost + offset) / epsilon)
    stress = {"cost_offset": offset, "epsilon": epsilon,
              "naive_kernel_zero_fraction": float(np.mean(naive_kernel == 0)),
              "maximum_plan_change": float(np.max(np.abs(base.plan - shifted.plan))),
              "primal_shift_error": float(shifted.primal - base.primal - offset),
              "base": diagnostics(base), "shifted": diagnostics(shifted)}
    if stress["maximum_plan_change"] > 1e-9 or abs(stress["primal_shift_error"]) > 1e-8:
        raise ArithmeticError("cost-offset invariance failed")
    all_results.extend((base, shifted))
    summary = {
        "schema_version": 1,
        "data": "Deterministic synthetic finite histograms; not sampled or real-world data.",
        "configuration": {"bins": bins, "grid_min": -1.5, "grid_max": 1.5,
                          "source_center": 0.0, "source_width": 0.35,
                          "epsilons": list(EPSILONS), "marginal_tolerance": 1e-11,
                          "gap_tolerance": 1e-9, "max_iterations": 10000,
                          "cost": "squared Euclidean", "kl_reference": "a outer b"},
        "software": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        "results": rows,
        "offset_stress": stress,
        "checks": {"number_of_solves": len(all_results),
                   "max_raw_marginal_error": max(r.raw_marginal_error for r in all_results),
                   "max_rounded_marginal_error": max(r.rounded_marginal_error for r in all_results),
                   "max_primal_dual_gap": max(r.gap for r in all_results),
                   "min_primal_dual_gap": min(r.gap for r in all_results),
                   "max_iterations_used": max(r.iterations for r in all_results)},
    }
    plans.update({"grid": grid.tolist(), "source": a.tolist()})
    return summary, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()
    summary, plans = run_audit(args.bins)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary", summary), ("couplings", plans)):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from .plotting import plot_audit
    plot_audit(summary, plans, args.output)
    print(json.dumps(summary["checks"], indent=2))
    print(f"Wrote summary, couplings, and two figures to {args.output}")


if __name__ == "__main__":
    main()
