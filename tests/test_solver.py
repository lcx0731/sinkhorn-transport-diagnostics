import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy

from transport_audit import ConvergenceError, round_plan, sinkhorn, squared_cost, unregularized_ot
from transport_audit.solver import marginal_error


@pytest.mark.parametrize("epsilon", [0.05, 0.3, 2.0])
def test_uniform_two_by_two_closed_form(epsilon):
    # Symmetry and the first-order condition give this solution analytically.
    off_diagonal = 0.5 / (1 + np.exp(1 / epsilon))
    expected = np.array([[0.5 - off_diagonal, off_diagonal],
                         [off_diagonal, 0.5 - off_diagonal]])
    result = sinkhorn([0.5, 0.5], [0.5, 0.5], [[0, 1], [1, 0]], epsilon)
    np.testing.assert_allclose(result.plan, expected, atol=1e-14)
    expected_objective = epsilon * (np.log(2) - np.log1p(np.exp(-1 / epsilon)))
    assert result.primal == pytest.approx(expected_objective, abs=1e-13)
    assert abs(result.gap) < 1e-13


def test_asymmetric_solution_against_independent_scalar_minimization():
    a, b = np.array([0.3, 0.7]), np.array([0.55, 0.45])
    cost, epsilon = np.array([[0.4, 1.2], [1.4, 0.1]]), 0.23

    def objective(t):
        plan = np.array([[t, a[0] - t], [b[0] - t, 1 - a[0] - b[0] + t]])
        return np.sum(plan * cost) + epsilon * np.sum(xlogy(plan, plan / np.outer(a, b)))

    independent = minimize_scalar(objective, bounds=(0, 0.3), method="bounded",
                                  options={"xatol": 1e-14})
    result = sinkhorn(a, b, cost, epsilon)
    assert independent.success
    assert result.plan[0, 0] == pytest.approx(independent.x, abs=2e-8)
    assert result.dual <= independent.fun + 1e-12
    assert result.primal >= independent.fun - 1e-12
    assert result.gap < 1e-9


@pytest.mark.parametrize("constant", [0.0, 0.4, 1000.0])
def test_constant_cost_gives_product_coupling(constant):
    a, b = np.array([0.2, 0.8]), np.array([0.1, 0.3, 0.6])
    result = sinkhorn(a, b, np.full((2, 3), constant), 0.1)
    np.testing.assert_allclose(result.plan, np.outer(a, b), atol=1e-12)
    assert abs(result.kl) < 1e-12
    assert result.primal == pytest.approx(constant, abs=1e-10)


def test_single_point_and_single_target_lp():
    result = sinkhorn([1], [1], [[3.5]], 0.2)
    assert result.primal == pytest.approx(3.5)
    assert result.plan[0, 0] == pytest.approx(1)
    lp = unregularized_ot([0.2, 0.8], [1], [[2], [3]])
    assert lp.cost == pytest.approx(2.8)


def test_rectangular_feasibility_and_dual_gauge():
    a, b = np.array([0.2, 0.8]), np.array([0.2, 0.3, 0.5])
    cost = squared_cost([-1, 1], [-2, 0, 2])
    result = sinkhorn(a, b, cost, 1.0)
    assert result.raw_marginal_error <= 1e-11
    assert marginal_error(result.plan, a, b) < 1e-13
    assert np.all(result.plan >= 0)
    assert abs(a @ result.f) < 1e-13
    raw_from_potentials = np.outer(a, b) * np.exp(result.f[:, None] + result.g - cost)
    np.testing.assert_allclose(raw_from_potentials, result.plan, atol=2e-11)
    lp = unregularized_ot(a, b, cost)
    assert result.transport_cost >= lp.cost - 1e-9
    assert result.primal >= result.transport_cost - 1e-13


def test_permutation_and_transpose_invariance():
    a, b = np.array([0.2, 0.8]), np.array([0.2, 0.3, 0.5])
    cost = squared_cost([-1, 1], [-2, 0, 2])
    result = sinkhorn(a, b, cost, 0.6)
    permutation = [2, 0, 1]
    reordered = sinkhorn(a[::-1], b[permutation], cost[::-1, permutation], 0.6)
    transposed = sinkhorn(b, a, cost.T, 0.6)
    np.testing.assert_allclose(reordered.plan, result.plan[::-1, permutation], atol=2e-11)
    np.testing.assert_allclose(transposed.plan, result.plan.T, atol=2e-11)
    assert result.primal == pytest.approx(transposed.primal, abs=1e-10)


def test_additive_offset_with_entire_naive_kernel_underflowing():
    a, b = [0.3, 0.7], [0.4, 0.6]
    cost = squared_cost([0, 1], [0, 2])
    epsilon, offset = 0.1, 1000.0
    assert np.all(np.exp(-(cost + offset) / epsilon) == 0)
    base = sinkhorn(a, b, cost, epsilon)
    shifted = sinkhorn(a, b, cost + offset, epsilon)
    np.testing.assert_allclose(shifted.plan, base.plan, atol=1e-10)
    assert shifted.primal - base.primal == pytest.approx(offset, abs=1e-9)


@pytest.mark.parametrize("initial", [np.zeros((2, 3)), np.ones((2, 3)),
                                      np.array([[0.2, 0, 0.9], [0.01, 0.7, 0]])])
def test_rounding_feasibility_including_empty_rows(initial):
    original = initial.copy()
    a, b = np.array([0.4, 0.6]), np.array([0.2, 0.3, 0.5])
    rounded = round_plan(initial, a, b)
    assert marginal_error(rounded, a, b) < 1e-13
    assert np.min(rounded) >= 0
    np.testing.assert_array_equal(initial, original)


def test_rounding_already_feasible_is_identity():
    plan = np.array([[0.25, 0.25], [0.1, 0.4]])
    np.testing.assert_allclose(round_plan(plan, [0.5, 0.5], [0.35, 0.65]), plan, atol=1e-15)


def test_nonconvergence_is_explicit():
    with pytest.raises(ConvergenceError, match="no convergence"):
        sinkhorn([0.1, 0.9], [0.8, 0.2], [[0, 5], [5, 0]], 0.1, max_iterations=1)


@pytest.mark.parametrize("a,b,cost,epsilon", [
    ([0.2, 0.5, 0.3], [0.2, 0.5, 0.3], squared_cost([-1, 0, 1], [-1, 0, 1]), 0.1),
    ([0.2, 0.5, 0.3], [0.2, 0.5, 0.3], squared_cost([-1, 0, 2], [-1, 0, 2]), 0.4),
    ([0.2, 0.8], [0.2, 0.3, 0.5], squared_cost([-1, 1], [-2, 0, 2]), 0.4),
])
def test_nearly_disconnected_problems_refuse_insufficient_budget(a, b, cost, epsilon):
    # Real limitation: log arithmetic prevents underflow, not slow convergence.
    # These cases also failed the default 10,000-step budget during development.
    with pytest.raises(ConvergenceError):
        sinkhorn(a, b, cost, epsilon, max_iterations=1000)


def test_lp_hand_computed_two_by_two():
    result = unregularized_ot([0.3, 0.7], [0.6, 0.4], [[0, 1], [1, 0]])
    np.testing.assert_allclose(result.plan, [[0.3, 0], [0.3, 0.4]], atol=1e-12)
    assert result.cost == pytest.approx(0.3)


def test_lp_against_independent_quantile_matching():
    x, a = np.array([-1, 0.3, 2.0]), np.array([0.2, 0.3, 0.5])
    y, b = np.array([-2, 0.2, 0.8, 3.0]), np.array([0.1, 0.2, 0.3, 0.4])
    # Integrate squared difference of the two inverse CDFs on their joint partition.
    breaks = np.unique(np.r_[0, np.cumsum(a), np.cumsum(b)])
    midpoint = 0.5 * (breaks[1:] + breaks[:-1])
    qx = x[np.searchsorted(np.cumsum(a), midpoint)]
    qy = y[np.searchsorted(np.cumsum(b), midpoint)]
    expected = np.sum(np.diff(breaks) * (qx - qy) ** 2)
    assert unregularized_ot(a, b, squared_cost(x, y)).cost == pytest.approx(expected, abs=1e-10)


def test_cost_in_multiple_dimensions():
    np.testing.assert_array_equal(squared_cost([[0, 0], [1, 2]], [[3, 4]]), [[25], [8]])


@pytest.mark.parametrize("a", [[0, 1], [-0.1, 1.1], [0.2, 0.2], [], [np.nan, 1], [[0.5, 0.5]]])
def test_invalid_weights(a):
    with pytest.raises(ValueError):
        sinkhorn(a, [1], [[1], [2]], 0.1)


@pytest.mark.parametrize("cost", [[[1]], [[-1, 0], [0, 1]], [[np.inf, 0], [0, 0]]])
def test_invalid_costs(cost):
    with pytest.raises(ValueError):
        sinkhorn([0.5, 0.5], [0.5, 0.5], cost, 0.1)


@pytest.mark.parametrize("options", [{"epsilon": 0}, {"epsilon": -1}, {"epsilon": np.nan},
                                     {"epsilon": np.inf}, {"tolerance": 0}, {"gap_tolerance": -1},
                                     {"max_iterations": 0}, {"max_iterations": 1.2},
                                     {"max_iterations": True}])
def test_invalid_solver_options(options):
    settings = {"epsilon": 0.1, **options}
    with pytest.raises(ValueError):
        sinkhorn([1], [1], [[0]], **settings)


@pytest.mark.parametrize("x,y", [([], [1]), ([np.nan], [1]), ([[1, 2]], [[1]]), ([1e300], [0])])
def test_invalid_point_arrays(x, y):
    with pytest.raises(ValueError):
        squared_cost(x, y)


@pytest.mark.parametrize("plan", [[[-1, 1], [1, 0]], [[np.nan, 0], [0, 1]], [[1]]])
def test_invalid_rounding_input(plan):
    with pytest.raises(ValueError):
        round_plan(plan, [0.5, 0.5], [0.5, 0.5])
