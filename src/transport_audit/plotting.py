"""Headless, reproducible figures with explicit objective labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_audit(summary, plans, output: Path):
    colors = {"identical": "#6b7280", "location_shift": "#087f8c", "location_and_scale": "#bd5d38"}
    labels = {"identical": "Identical", "location_shift": "Location shift", "location_and_scale": "Location + scale"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        for case, color in colors.items():
            rows = [r for r in summary["results"] if r["case"] == case]
            eps = [r["epsilon"] for r in rows]
            axes[0].semilogx(eps, [r["cross"]["primal"] for r in rows], "o-", color=color, label=labels[case])
            axes[1].semilogx(eps, [r["divergence"] for r in rows], "o-", color=color, label=labels[case])
            axes[1].axhline(rows[0]["lp_cost"], color=color, linestyle=":", alpha=0.8)
        axes[0].set(title="Raw regularized OT has nonzero self-cost", ylabel="Transport cost + epsilon × KL")
        axes[1].set(title="Subtracting self-costs removes that baseline", ylabel="Sinkhorn divergence")
        for ax in axes:
            ax.set_xlabel("Regularization epsilon (log scale)")
            ax.grid(alpha=0.15)
        axes[0].legend(frameon=False)
        axes[1].plot([], [], ":", color="black", label="Unregularized LP cost (reference)")
        axes[1].legend(frameon=False, fontsize=8, loc="center left")
        fig.suptitle("Finite synthetic histograms • squared Euclidean cost • KL relative to a ⊗ b", fontsize=12)
        fig.savefig(output / "self-cost-correction.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained")
        grid = np.array(plans["grid"])
        axes[0].plot(grid, plans["source"], color="#087f8c", label="Source")
        axes[0].plot(grid, plans["target"], color="#bd5d38", label="Target")
        axes[0].set(title="Location + scale case", xlabel="Support value", ylabel="Probability mass")
        axes[0].legend(frameon=False)
        maximum = max(np.max(plans[key]) for key in ("lp", "0.03", "1.0"))
        half_step = (grid[1] - grid[0]) / 2
        extent = [grid[0] - half_step, grid[-1] + half_step] * 2
        for ax, key, title in zip(axes[1:], ("lp", "0.03", "1.0"),
                                  ("Unregularized LP", "Log-Sinkhorn: epsilon = 0.03", "Log-Sinkhorn: epsilon = 1.0")):
            im = ax.imshow(plans[key], origin="lower", extent=extent, vmin=0, vmax=maximum,
                           cmap="magma", interpolation="nearest")
            ax.set(title=title, xlabel="Target support", ylabel="Source support")
        fig.colorbar(im, ax=list(axes[1:]), label="Coupling mass (common linear scale)", shrink=0.75)
        fig.savefig(output / "transport-plans.png", dpi=160)
        plt.close(fig)
