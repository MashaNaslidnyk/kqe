import itertools
from functools import partial
from typing import List, Tuple

import jax.numpy as jnp
from jax import Array, jit, random, vmap

from kqe.kernels import KernelLike


def create_indices(N: int, R: int) -> Tuple[List[int], List[int]]:
    """
    Taken from https://github.com/antoninschrab/agginc/blob/main/agginc/jax.py

    Return lists of indices of R superdiagonals of N x N matrix

    This function can be modified to compute any type of incomplete U-statistic.

    Returns K/2 * (2N - K - 1) indices for K superdiagonals of an N x N matrix.
    """

    index_X = list(
        itertools.chain(*[[i for i in range(N - r)] for r in range(1, R + 1)])
    )
    index_Y = list(
        itertools.chain(*[[i + r for i in range(N - r)] for r in range(1, R + 1)])
    )
    return index_X, index_Y


def core_function(x1, y1, x2, y2, kernel_fn: KernelLike) -> Array:
    """Core function for the MMD estimators."""
    vectorised_kernel_fn = vmap(lambda xx1, xx2: kernel_fn(xx1, xx2))
    return (
        vectorised_kernel_fn(x1, x2)
        + vectorised_kernel_fn(y1, y2)
        - vectorised_kernel_fn(x1, y2)
        - vectorised_kernel_fn(x2, y1)
    )


@jit
def mmd_squared_V_statistic(X: Array, Y: Array, kernel_fn: KernelLike) -> Array:
    """Compute the V-statistic MMD estimator between two sample sets."""

    def gram_matrix(X1: Array, X2: Array) -> Array:
        """Compute the Gram matrix of two sets of vectors."""
        return vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(X2))(X1)

    K_xx = gram_matrix(X, X)
    K_yy = gram_matrix(Y, Y)
    K_xy = gram_matrix(X, Y)
    K_yx = gram_matrix(Y, X)

    return (K_xx + K_yy - K_xy - K_yx).mean()


@jit
def mmd_linear_estimator(X: Array, Y: Array, kernel_fn: KernelLike) -> Array:
    """Compute the linear MMD approximation from gretton2012 between two sample sets."""

    assert X.shape == Y.shape, "The two samples should have the same shape."

    m2 = int(X.shape[0] / 2)
    res = jnp.mean(
        core_function(X[:m2], Y[:m2], X[m2 : 2 * m2], Y[m2 : 2 * m2], kernel_fn)
    )

    return res


@partial(jit, static_argnames=["num_diagonals"])
def mmd_multi_diagonal_estimator(
    X: Array, Y: Array, kernel_fn: KernelLike, num_diagonals: int = 1
) -> Array:
    """
    Compute the MMD approximation based on multiple diagonals, of Schrab et al,
        "Efficient Aggregated Kernel Tests using Incomplete U-statistics", Eq 8.

    :param num_diagonals: Number of diagonals to use.
    """
    assert X.shape == Y.shape, "The two samples should have the same shape."
    assert (
        0 < num_diagonals <= X.shape[0]
    ), "The number of diagonals should be between 0 and the sample size."
    # TODO: Fix the assertion above to work with jit. There is something in chex

    ind1, ind2 = create_indices(X.shape[0], num_diagonals)
    ind1 = jnp.array(ind1)
    ind2 = jnp.array(ind2)

    res = jnp.mean(core_function(X[ind1], Y[ind1], X[ind2], Y[ind2], kernel_fn))

    return res


@jit
def mmd_squared_U_statistic(X: Array, Y: Array, kernel_fn: KernelLike) -> Array:
    """Compute the U-statistic MMD estimator between two sample sets."""

    n = X.shape[0]
    m = Y.shape[0]

    K_xx = vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(X))(X)
    K_xx = K_xx - jnp.diag(jnp.diag(K_xx))

    K_yy = vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(Y))(Y)
    K_yy = K_yy - jnp.diag(jnp.diag(K_yy))

    K_xy = vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(X))(Y)
    term1 = K_xx.sum() / (n * (n - 1))
    term2 = -2 * K_xy.sum() / (n * m)
    term3 = K_yy.sum() / (m * (m - 1))

    return term1 + term2 + term3


@partial(jit, static_argnames=["num_nystrom_points"])
def nystrom_mmd_sq(
    X: Array,
    Y: Array,
    kernel_fn: KernelLike,
    num_nystrom_points: int,
    key_test: random.PRNGKey,
) -> Array:
    """
    Compute Nystrom approximation of MMD, of Chatalic et al,
        "Nyström Kernel Mean Embeddings", Eq 5.

    :param num_nystrom_points: Number of points m.
    :param key: JAX PRNG key for sampling m points.
    """

    def nystrom_features():
        Knm = vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(Z[inds]))(Z)
        Km = Knm[inds, :]  # Shape: [k, k]

        U, S, Vt = jnp.linalg.svd(Km)
        S = jnp.maximum(S, 1e-12)  # Ensures numerical stability for SVD inversion.
        W = jnp.dot(U / S, Vt)  # Shape: [k, k]

        return Knm @ W.T

    n = X.shape[0]
    m = num_nystrom_points

    Z = jnp.concatenate([X, Y])

    inds = random.choice(key_test, n, (m,), replace=False)
    psi_Z = nystrom_features()

    X_ind = jnp.concatenate((jnp.ones(n), jnp.zeros(n)))
    Y_ind = 1 - X_ind

    bar_Z_B_piX = (1 / m) * X_ind @ psi_Z
    bar_Z_B_piY = (1 / n) * Y_ind @ psi_Z
    T = bar_Z_B_piX - bar_Z_B_piY
    V = jnp.sum(T**2)

    return V


@partial(jit, static_argnames=("num_random_me_points",))
def me_test_statistic(
    X: Array,
    Y: Array,
    kernel_fn: KernelLike,
    num_random_me_points: int,
    key_test: random.PRNGKey,
) -> Array:
    """
    Compute the analytic mean embeddings (ME) test statistic of Chwialkowski et al.,
        "Fast two-sample testing with analytic representations of probability measures", Eq 10.

    :param num_random_me_points: Number of random points T to use.
    :param key: JAX PRNG key for sampling random points.
    """
    w = random.normal(key_test, shape=(num_random_me_points, X.shape[1]))

    kXT = vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(X))(w)
    kYT = vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(Y))(w)

    diff = jnp.mean(jnp.mean(kXT - kYT, axis=1) ** 2)

    return diff


if __name__ == "__main__":
    from kqe.kernels import GaussianKernel

    X = random.normal(random.PRNGKey(0), (1000,))
    Y = random.normal(random.PRNGKey(1), (1000,))

    res1 = mmd_linear_estimator(X, Y, GaussianKernel(1.0))
    print(res1)

    res2 = mmd_squared_U_statistic(X, Y, GaussianKernel(1.0))
    print(res2)
