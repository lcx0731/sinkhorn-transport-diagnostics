import numpy as np
import pytest

from transport_audit import sinkhorn_divergence


@pytest.mark.parametrize("epsilon", [0.03, 0.1, 0.3, 1.0])
def test_identical_distribution_zero_after_correction(epsilon):
    result = sinkhorn_divergence([-0.3, 0, 0.3], [0.2, 0.5, 0.3],
                                 [-0.3, 0, 0.3], [0.2, 0.5, 0.3], epsilon)
    assert result.cross.primal > 0
    assert result.value == pytest.approx(0, abs=1e-14)
    assert result.lower <= 1e-12
    assert result.upper >= -1e-12


def test_divergence_symmetry_and_propagated_bracket():
    x, a = [-0.5, 0, 1], [0.2, 0.5, 0.3]
    y, b = [0, 0.5], [0.7, 0.3]
    result = sinkhorn_divergence(x, a, y, b, 0.4)
    reverse = sinkhorn_divergence(y, b, x, a, 0.4)
    assert result.value >= 0
    assert result.value == pytest.approx(reverse.value, abs=1e-9)
    assert result.lower - 1e-12 <= result.value <= result.upper + 1e-12
    expected_width = result.cross.gap + 0.5 * (result.self_a.gap + result.self_b.gap)
    assert result.upper - result.lower == pytest.approx(expected_width, abs=1e-14)


def test_translation_preserves_shape_and_has_known_divergence():
    x, a = np.array([-1, 0, 1]), np.array([0.2, 0.5, 0.3])
    # Same weights translated by d: cross objective = self objective + d^2.
    # This exact identity does not apply to truncated histograms on a fixed grid.
    result = sinkhorn_divergence(x, a, x + 0.4, a, 0.3)
    assert result.value == pytest.approx(0.4**2, abs=1e-9)
    assert result.lower <= 0.4**2 + 1e-12
    assert result.upper >= 0.4**2 - 1e-12
