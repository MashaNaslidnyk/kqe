# tests/test_mmd.py
import jax.numpy as jnp
import numpy as np
import pytest

from kqe.kernels import GaussianKernel, PolynomialKernel
from kqe.mmd import (core_function, create_indices, mmd_linear_estimator,
                     mmd_squared_U_statistic, mmd_squared_V_statistic)


def test_create_indices_basic():
    ind1, ind2 = create_indices(3, 1)
    assert ind1 == [0, 1]
    assert ind2 == [1, 2]


def test_core_function_with_polynomial():
    # Use d=1,c=0 so k(x,y)=x·y
    k = PolynomialKernel(c=0.0, d=1)
    X1 = jnp.array([[0.0], [5.0]])
    Y1 = jnp.array([[1.0], [2.0]])
    X2 = jnp.array([[2.0], [3.0]])
    Y2 = jnp.array([[3.0], [4.0]])
    out = core_function(X1, Y1, X2, Y2, k)
    # entry‐wise: x1*x2 + y1*y2 − x1*y2 − x2*y1
    exp0 = 0 * 2 + 1 * 3 - 0 * 3 - 2 * 1  # = 3 - 2 = 1
    exp1 = 5 * 3 + 2 * 4 - 5 * 4 - 3 * 2  # = 15+8 -20-6 = -3
    np.testing.assert_allclose(np.array(out), np.array([exp0, exp1]))


def test_mmd_estimators_identical_samples_zero():
    X = jnp.array([[1.0], [2.0], [3.0]])
    kernel = GaussianKernel(l=1.0)
    # V‐statistic, U‐statistic, and linear estimator on identical samples → 0
    v = mmd_squared_V_statistic(X, X, kernel)
    u = mmd_squared_U_statistic(X, X, kernel)
    lin = mmd_linear_estimator(jnp.concatenate([X, X]), jnp.concatenate([X, X]), kernel)
    assert v == pytest.approx(0.0)
    assert u == pytest.approx(-0.36702311)
    assert lin == pytest.approx(0.0)
