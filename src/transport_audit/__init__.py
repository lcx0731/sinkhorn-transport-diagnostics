"""Small, inspectable numerical optimal-transport experiments."""

from .solver import ConvergenceError, round_plan, sinkhorn, squared_cost, unregularized_ot
from .divergence import sinkhorn_divergence

__all__ = [
    "ConvergenceError", "round_plan", "sinkhorn", "squared_cost",
    "unregularized_ot", "sinkhorn_divergence",
]
