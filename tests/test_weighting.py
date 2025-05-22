# tests/test_weighting.py
import jax.numpy as jnp
import numpy as np
import pytest

from kqe.weighting import (get_weights, shape_flat, shape_reverse_triangle,
                           shape_slope_down, shape_slope_up, shape_triangle)


def test_shape_slope_up():
    assert shape_slope_up(0.0, 0.3) == pytest.approx(0.3)
    assert shape_slope_up(1.0, 0.3) == pytest.approx(1.0)
    assert shape_slope_up(0.5, 0.3) == pytest.approx(0.3 + (1 - 0.3) * 0.5)


def test_shape_slope_down():
    assert shape_slope_down(0.0, 0.4) == pytest.approx(1.0)
    assert shape_slope_down(1.0, 0.4) == pytest.approx(0.4)
    assert shape_slope_down(0.5, 0.4) == pytest.approx(1.0 - (1 - 0.4) * 0.5)


def test_shape_triangle_and_reverse():
    t = shape_triangle(jnp.array([0.0, 0.5, 1.0]), 0.2)
    rt = shape_reverse_triangle(jnp.array([0.0, 0.5, 1.0]), 0.2)
    np.testing.assert_allclose(np.array(t), np.array([0.2, 1.0, 0.2]))
    np.testing.assert_allclose(np.array(rt), np.array([1.0, 0.2, 1.0]))


def test_shape_flat():
    arr = jnp.array([0.0, 0.3, 1.0])
    out = shape_flat(arr, ratio=0.9)
    assert np.all(out == 1.0)


def test_get_weights_slope_up():
    w = get_weights(2, shape="slope_up", ratio=0.5)
    # raw = [0.5,1.0] → normalized to [1/3,2/3]
    np.testing.assert_allclose(np.array(w), np.array([1 / 3, 2 / 3]))


def test_get_weights_flat():
    n = 5
    w = get_weights(n, shape="flat", ratio=0.7)
    np.testing.assert_allclose(np.array(w), np.ones(n) / n)
