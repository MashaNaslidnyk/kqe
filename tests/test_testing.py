# tests/test_testing.py
import jax.numpy as jnp
import pytest
from jax import random

from kqe.testing import permutation_test, shuffle_cut_and_compute_distance


def test_shuffle_cut_and_compute_distance_simple():
    joint = jnp.array([1, 2, 3, 4])
    perm = jnp.array([0, 1, 2, 3])

    # split in half → left=[1,2], right=[3,4]
    def ts(l, r):
        return jnp.sum(l) - jnp.sum(r)

    out = shuffle_cut_and_compute_distance(joint, perm, ts)
    assert out == pytest.approx((1 + 2) - (3 + 4))


def test_permutation_test_identical_returns_zero_and_pval_one():
    key = random.PRNGKey(0)
    X = jnp.array([1.0, 2.0, 3.0, 4.0])
    Y = X

    def ts(a, b):
        return jnp.abs(jnp.mean(a) - jnp.mean(b))

    # binary decision
    decision = permutation_test(
        key,
        X,
        Y,
        ts,
        num_permutations=10,
        batch_size=5,
        level=0.05,
        return_p_val=False,
    )
    assert decision == 0
    # p‐value
    pval = permutation_test(
        key,
        X,
        Y,
        ts,
        num_permutations=10,
        batch_size=5,
        level=0.05,
        return_p_val=True,
    )
    assert pval == pytest.approx(1.0)
