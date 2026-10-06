"""Balanced finite OT, with KL relative to the product of the marginals.

This is an original implementation of standard algorithms, not a new OT method.
Rounding follows Altschuler, Weed & Rigollet (2017), Algorithm 2.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import logsumexp, xlogy

Array = NDArray[np.float64]


class ConvergenceError(RuntimeError):
    """Requested numerical accuracy was not reached; no result is returned."""


@dataclass(frozen=True)
class TransportResult:
    plan: Array
    f: Array
    g: Array
    iterations: int
    raw_marginal_error: float
    rounded_marginal_error: float
    transport_cost: float
    kl: float
    primal: float
    dual: float
    gap: float


@dataclass(frozen=True)
class LPResult:
    plan: Array
    cost: float
    marginal_error: float


def weights(value: ArrayLike) -> Array:
    """Validate positive unit mass; normalize only <=1e-12 sum roundoff."""
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("weights must be a nonempty finite vector")
    if np.any(value <= 0):
        raise ValueError("weights must be positive; remove zero-mass support points")
    total = float(value.sum())
    if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
        raise ValueError("weights must sum to one")
    return value / total


def problem(a: ArrayLike, b: ArrayLike, cost: ArrayLike) -> tuple[Array, Array, Array]:
    a, b = weights(a), weights(b)
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (a.size, b.size) or not np.isfinite(cost).all():
        raise ValueError("cost must be a finite matrix matching the marginals")
    if np.any(cost < 0):
        raise ValueError("cost must be nonnegative")
    return a, b, cost


def squared_cost(x: ArrayLike, y: ArrayLike) -> Array:
    """Squared Euclidean costs for n-by-d points (vectors mean d=1)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or y.shape[0] == 0
            or x.shape[1] == 0 or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("points must be finite nonempty arrays with matching dimensions")
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    if not np.isfinite(cost).all():
        raise ValueError("squared distances overflow; rescale the points")
    return cost


def marginal_error(plan: Array, a: Array, b: Array) -> float:
    """Maximum absolute entrywise marginal residual, not relative error."""
    return float(max(np.max(np.abs(plan.sum(axis=1) - a)),
                     np.max(np.abs(plan.sum(axis=0) - b))))


def round_plan(plan: ArrayLike, a: ArrayLike, b: ArrayLike) -> Array:
    """Scale down rows, then columns; fill deficits with a rank-one matrix.

    Feasible in exact arithmetic. Tiny floating-point deficits can disagree,
    so the returned marginal residual is checked explicitly (<=1e-13).
    This also handles a zero initial matrix without dividing by zero.
    """
    a, b = weights(a), weights(b)
    plan = np.asarray(plan, dtype=float).copy()
    if (plan.shape != (a.size, b.size) or not np.isfinite(plan).all()
            or np.any(plan < 0)):
        raise ValueError("plan must be finite, nonnegative, and match the marginals")
    for axis, target in ((1, a), (0, b)):
        mass = plan.sum(axis=axis)
        if not np.isfinite(mass).all():
            raise ValueError("plan mass overflows")
        factor = np.divide(np.minimum(mass, target), mass,
                           out=np.ones_like(mass), where=mass > 0)
        plan *= factor[:, None] if axis == 1 else factor[None, :]
    da, db = a - plan.sum(axis=1), b - plan.sum(axis=0)
    if min(da.min(), db.min()) < -1e-13:
        raise ArithmeticError("rounding produced a negative marginal deficit")
    da, db = np.maximum(da, 0), np.maximum(db, 0)
    if da.sum() > 0:
        # Divide first to avoid underflow in the outer product for tiny deficits.
        plan += np.outer(da / da.sum(), db)
    if marginal_error(plan, a, b) > 1e-13:
        raise ArithmeticError("rounded plan is not numerically feasible")
    return plan


def _result(raw: Array, a: Array, b: Array, cost: Array, epsilon: float,
            f: Array, g: Array, log_plan: Array, iteration: int) -> TransportResult:
    plan = round_plan(raw, a, b)
    log_product = np.log(a)[:, None] + np.log(b)[None, :]
    # Generalized KL: sum p log(p/q) - p + q, with 0 log 0 = 0.
    kl = float(np.sum(xlogy(plan, plan) - plan * log_product) - plan.sum() + 1)
    linear = float(np.sum(plan * cost))
    primal = linear + epsilon * kl
    dual = float(a @ f + b @ g - epsilon * np.expm1(logsumexp(log_plan)))
    return TransportResult(plan, f.copy(), g.copy(), iteration,
                           marginal_error(raw, a, b), marginal_error(plan, a, b),
                           linear, kl, primal, dual, primal - dual)


def sinkhorn(a: ArrayLike, b: ArrayLike, cost: ArrayLike, epsilon: float, *,
             tolerance: float = 1e-11, gap_tolerance: float = 1e-9,
             max_iterations: int = 10000) -> TransportResult:
    """Log-domain alternating dual updates, with explicit convergence checks.

    Minimize <C,P> + epsilon KL(P || a outer b), subject to P1=a, P.T1=b.
    Check both raw marginal residual and rounded primal-minus-dual gap.
    Negative gaps within a scale-aware float64 roundoff allowance are retained,
    never hidden by clipping. These are numerical diagnostics, not rigorous
    interval-arithmetic certificates. Dense memory and work per iteration: O(nm).
    """
    a, b, cost = problem(a, b, cost)
    for value, name in ((epsilon, "epsilon"), (tolerance, "tolerance"),
                        (gap_tolerance, "gap_tolerance")):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations <= 0):
        raise ValueError("max_iterations must be a positive integer")
    loga, logb = np.log(a), np.log(b)
    f, g = np.zeros_like(a), np.zeros_like(b)
    error = float("inf")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for iteration in range(1, max_iterations + 1):
                f = -epsilon * logsumexp(logb[None, :] + (g[None, :] - cost) / epsilon, axis=1)
                g = -epsilon * logsumexp(loga[:, None] + (f[:, None] - cost) / epsilon, axis=0)
                # Gauge f -> f-k, g -> g+k leaves the implied plan unchanged.
                center = a @ f
                f, g = f - center, g + center
                if iteration % 10 == 0 or iteration == max_iterations:
                    log_plan = loga[:, None] + logb[None, :] + (f[:, None] + g[None, :] - cost) / epsilon
                    raw = np.exp(log_plan)
                    error = marginal_error(raw, a, b)
                    if error <= tolerance:
                        result = _result(raw, a, b, cost, epsilon, f, g, log_plan, iteration)
                        scale = 1 + max(abs(result.primal), abs(result.dual))
                        slack = 128 * np.finfo(float).eps * scale
                        if not np.isfinite([result.primal, result.dual, result.kl]).all():
                            raise ArithmeticError("nonfinite objective")
                        if result.gap < -slack:
                            raise ArithmeticError("dual exceeds rounded primal beyond roundoff")
                        if result.gap <= gap_tolerance:
                            return result
    except FloatingPointError as exc:
        raise ArithmeticError("numerical range exceeded; rescale costs or increase epsilon") from exc
    raise ConvergenceError(f"no convergence after {max_iterations} iterations; marginal error={error:.3g}")


def unregularized_ot(a: ArrayLike, b: ArrayLike, cost: ArrayLike) -> LPResult:
    """Numerical LP reference for small problems, without entropic penalty."""
    a, b, cost = problem(a, b, cost)
    n, m = cost.shape
    rows = sparse.kron(sparse.eye(n), np.ones((1, m)), format="csr")
    cols = sparse.kron(np.ones((1, n)), sparse.eye(m), format="csr")
    # The last column equality is redundant when the total mass is one.
    constraints = sparse.vstack((rows, cols[:-1]), format="csr")
    solution = linprog(cost.ravel(), A_eq=constraints, b_eq=np.r_[a, b[:-1]],
                       bounds=(0, None), method="highs",
                       options={"primal_feasibility_tolerance": 1e-9,
                                "dual_feasibility_tolerance": 1e-9})
    if not solution.success:
        raise ConvergenceError(f"LP failed: {solution.message}")
    plan = solution.x.reshape(n, m)
    error = marginal_error(plan, a, b)
    if error > 1e-8 or np.min(plan) < -1e-10:
        raise ArithmeticError("LP returned an infeasible plan")
    return LPResult(plan, float(np.sum(plan * cost)), error)
