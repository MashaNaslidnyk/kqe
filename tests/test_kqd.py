# tests/test_kqd.py
import jax.numpy as jnp
import pytest

from kqe.kernels import GaussianKernel
from kqe.kqd import ekqd, ekqd_centered, supkqd


def test_supkqd_identical_samples_zero():
    X = jnp.array([[1.0], [2.0], [3.0]])
    # defaults: mus=None, num_mus=None, nu_shape='flat', nu_ratio=1.0, normalise=True, p=2
    out = supkqd(X, X, num_projections=5, kernel_fn=GaussianKernel(l=1.0))
    assert out == pytest.approx(0.0)


def test_ekqd_identical_samples_zero():
    X = jnp.array([[4.0], [5.0]])
    out = ekqd(X, X, num_projections=3, kernel_fn=GaussianKernel(l=2.0))
    assert out == pytest.approx(0.0)


def test_ekqd_centered_identical_samples_zero():
    X = jnp.array([[7.0], [9.0]])
    out = ekqd_centered(X, X, num_projections=2, kernel_fn=GaussianKernel(l=1.0))
    assert out == pytest.approx(0.0)


def test_supkqd_diff_samples():
    X = jnp.array([[1.0], [2.0], [3.0]])
    Y = jnp.array([[4.0], [5.0], [6.0]])
    # defaults: mus=None, num_mus=None, nu_shape='flat', nu_ratio=1.0, normalise=True, p=2
    out = supkqd(X, Y, num_projections=5, kernel_fn=GaussianKernel(l=1.0))
    assert out == pytest.approx(0.6356133819)


def test_ekqd_diff_samples():
    X = jnp.array([[4.0], [5.0]])
    Y = jnp.array([[6.0], [7.0]])
    out = ekqd(X, Y, num_projections=3, kernel_fn=GaussianKernel(l=2.0))
    assert out == pytest.approx(0.2308304459)


def test_ekqd_centered_diff_samples():
    X = jnp.array([[7.0], [9.0]])
    Y = jnp.array([[8.0], [10.0]])
    out = ekqd_centered(X, Y, num_projections=2, kernel_fn=GaussianKernel(l=1.0))
    assert out == pytest.approx(0.5697478056)


def test_ekqd_diff_samples_diff_pars():
    X = jnp.array([[1.0], [2.0], [3.0]])
    Y = jnp.array([[4.0], [5.0], [6.0]])
    out = ekqd(
        X,
        Y,
        num_projections=3,
        kernel_fn=GaussianKernel(l=2.0),
        normalise=False,
        p=1,
        nu_shape="triangle",
        nu_ratio=0.0,
    )
    assert out == pytest.approx(1.3815598488)
