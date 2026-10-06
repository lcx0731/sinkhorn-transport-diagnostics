"""Self-cost correction with propagated numerical primal/dual brackets."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .solver import TransportResult, sinkhorn, squared_cost


@dataclass(frozen=True)
class DivergenceResult:
    value: float
    lower: float
    upper: float
    cross: TransportResult
    self_a: TransportResult
    self_b: TransportResult


def sinkhorn_divergence(x: ArrayLike, a: ArrayLike, y: ArrayLike, b: ArrayLike,
                        epsilon: float, **solver_options) -> DivergenceResult:
    """OT_eps(a,b) - (OT_eps(a,a)+OT_eps(b,b))/2 for squared costs.

    The value uses rounded primal objectives. The lower/upper bracket combines
    three primal/dual pairs. Float64 roundoff remains; no negative clipping.
    This is not claimed to satisfy the triangle inequality.
    """
    cross = sinkhorn(a, b, squared_cost(x, y), epsilon, **solver_options)
    self_a = sinkhorn(a, a, squared_cost(x, x), epsilon, **solver_options)
    self_b = sinkhorn(b, b, squared_cost(y, y), epsilon, **solver_options)
    value = cross.primal - 0.5 * (self_a.primal + self_b.primal)
    lower = cross.dual - 0.5 * (self_a.primal + self_b.primal)
    upper = cross.primal - 0.5 * (self_a.dual + self_b.dual)
    return DivergenceResult(value, lower, upper, cross, self_a, self_b)
