# tests/test_kernels.py
import jax.numpy as jnp
import numpy as np
import pytest

from kqe.kernels import (GaussianKernel, PolynomialKernel,
                         PolynomialNormalisedKernel, compute_median_heuristic)


def test_gaussian_kernel_identity():
    k = GaussianKernel(l=2.0)
    x = jnp.array([1.0, 2.0, 3.0])
    # identical inputs → zero distance → exp(0) = 1
    assert k(x, x) == pytest.approx(1.0)


def test_gaussian_kernel_distance():
    k = GaussianKernel(l=1.0)
    x = jnp.array([0.0])
    y = jnp.array([2.0])
    # squared distance = (0−2)^2 = 4 → −0.5*4 = −2 → exp(−2)
    assert k(x, y) == pytest.approx(np.exp(-2.0))


def test_polynomial_kernel_basic():
    k = PolynomialKernel(c=1.0, d=3)
    x = jnp.array([1.0, 0.0])
    y = jnp.array([0.0, 1.0])
    # dot = 0 → (0 + 1)**3 = 1
    assert k(x, y) == pytest.approx(1.0)


def test_polynomial_normalised_kernel_identity():
    k = PolynomialNormalisedKernel(c=0.0, d=2)
    x = jnp.array([3.0, 4.0])  # ‖x‖^2 = 25
    # numerator = (x⋅x + 0)**2 = 25**2 = 625
    # denominator = (25**1)*(25**1) = 625 → ratio = 1
    assert k(x, x) == pytest.approx(1.0)


def test_compute_median_heuristic_two_points():
    x = jnp.array([[0.0], [2.0]])
    # pairwise squared‐half‐distances: [[0, 2], [2, 0]] → median = 1 → sqrt(1) = 1
    assert compute_median_heuristic(x) == pytest.approx(1.0)


def test_compute_median_heuristic_with_two_sets():
    x1 = jnp.array([[0.0], [4.0]])
    x2 = jnp.array([[2.0]])
    # combined = [0,4,2] → distances matrix /2 → flatten → compute median → sqrt
    d = np.array([[0, 8, 2], [8, 0, 2], [2, 2, 0]]) / 2.0
    expected = np.sqrt(np.median(d.flatten()))
    assert compute_median_heuristic(x1, x2) == pytest.approx(expected)


def test_compute_median_heuristic_with_two_sets():
    x1 = jnp.array([[0.0], [4.0]])
    x2 = jnp.array([[2.0]])
    # points [0, 4, 2] → squared distances/2 gives:
    # [[0, 16/2, 4/2],
    #  [16/2, 0, 4/2],
    #  [4/2, 4/2, 0]]
    # = [[0, 8, 2], [8, 0, 2], [2, 2, 0]]
    d = np.array([[0, 8, 2], [8, 0, 2], [2, 2, 0]])
    expected = np.sqrt(np.median(d.flatten()))
    assert compute_median_heuristic(x1, x2) == pytest.approx(expected)


def test_compute_median_heuristic_with_two_sets():
    x1 = jnp.array([[0.0], [4.0]])
    x2 = jnp.array([[2.0]])

    xs = np.concatenate([x1, x2])
    d = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            d[i, j] = (xs[i, 0] - xs[j, 0]) ** 2 / 2

    expected = np.sqrt(np.median(d.flatten()))
    assert compute_median_heuristic(x1, x2) == pytest.approx(expected)
