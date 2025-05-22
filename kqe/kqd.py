from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, jit, random, vmap

from kqe.logging_utils import info
from kqe.mmd import mmd_squared_V_statistic
from kqe.weighting import get_weights


@jit
def rkhs_function(
    lmbda: Array,  # Shape: (m,)
    kernel_fn: Callable[[Array, Array], Array],  # Kernel function
    X: Array,  # Shape: (m, d)
    arg: Array,  # Shape: (n, d)
) -> Array:  # Output shape: (n,)
    """Compute the RKHS function f(x) = sum_i lambda[i] * kernel_fn(arg, X[i])."""
    kernel_values = vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(X))(arg)
    return jnp.dot(kernel_values, lmbda)


@jit
def rkhs_norm_sq(
    lmbda: Array,  # Shape: (m,)
    kernel_fn: Callable[[Array, Array], Array],  # Kernel function
    X: Array,  # Shape: (m, d)
) -> Array:
    """Compute the RKHS norm squared."""
    gram_matrix = vmap(lambda x1: vmap(kernel_fn, in_axes=(None, 0))(x1, X))(X)
    return lmbda @ gram_matrix @ lmbda


def get_mus(X, Y, mus, num_mus, key):
    """Prepare the mus values."""
    combined = jnp.concatenate([X, Y])
    if mus is None:
        if num_mus is None or num_mus >= combined.shape[0]:
            return combined
        indices = random.choice(key, combined.shape[0], shape=(num_mus,), replace=False)
        return combined[indices]

    return mus


@partial(jit, static_argnames=["normalise", "nu_shape", "centered"])
def compute_tau_power_p(
    key,
    kernel_fn,
    mus,
    X,
    Y,
    nu_shape,
    nu_ratio,
    normalise,
    p,
    centered=False,
    mmd_sq=None,
):
    """Compute the power-p directional differences."""
    if centered and mmd_sq is None:
        mmd_sq = mmd_squared_V_statistic(X, Y, kernel_fn)

    coeffs_i = random.normal(key, (mus.shape[0],))
    f_i_X = rkhs_function(coeffs_i, kernel_fn, mus, X)
    f_i_Y = rkhs_function(coeffs_i, kernel_fn, mus, Y)

    norm_f_i_sq = rkhs_norm_sq(coeffs_i, kernel_fn, mus) + 1e-6 if normalise else 1.0

    if centered:
        power_p_diff = jnp.power(
            jnp.abs(
                (
                    (jnp.sort(f_i_X) - jnp.sort(f_i_Y)) ** 2
                    - (jnp.mean(f_i_X) - jnp.mean(f_i_Y)) ** 2
                )
                / norm_f_i_sq
                + mmd_sq
            ),
            p / 2,
        )
    else:
        power_p_diff = jnp.power(jnp.abs(jnp.sort(f_i_X) - jnp.sort(f_i_Y)), p)
        
    weights = get_weights(f_i_X.shape[0], nu_shape, nu_ratio)
    tau_power_p = jnp.dot(weights, power_p_diff)

    return tau_power_p


@partial(
    jit,
    static_argnames=["num_projections", "num_mus", "normalise", "nu_shape", "metric"],
)
def compute_distance(
    X,
    Y,
    num_projections,
    kernel_fn,
    mus=None,
    num_mus=None,
    nu_shape="flat",
    nu_ratio=1.0,
    normalise=True,
    p=2,
    metric="average",
):
    """
    Generalised function to compute e-KQD, e-KQD-Centered, and sup-KQD.
    """
    info(
        f"Compiled {metric} distance with {num_projections=}, {num_mus=}, {normalise=}"
    )

    keys = random.split(random.PRNGKey(0), num_projections + 1)
    mus = get_mus(X, Y, mus, num_mus, keys[-1])
    centered = metric == "centered"

    mmd_sq = mmd_squared_V_statistic(X, Y, kernel_fn) if centered else None

    projections = vmap(
        lambda key: compute_tau_power_p(
            key,
            kernel_fn=kernel_fn,
            mus=mus,
            X=X,
            Y=Y,
            nu_shape=nu_shape,
            nu_ratio=nu_ratio,
            normalise=normalise,
            p=p,
            centered=centered,
            mmd_sq=mmd_sq,
        )
    )(keys)

    if metric == "max":
        return jnp.max(projections)

    return jnp.power(jnp.mean(projections), 1 / p)


@partial(jit, static_argnames=["num_projections", "num_mus", "normalise", "nu_shape"])
def ekqd(
    X,
    Y,
    num_projections,
    kernel_fn,
    mus=None,
    num_mus=None,
    nu_shape="flat",
    nu_ratio=1.0,
    normalise=True,
    p=2,
):
    """Compute the e-KQD distance."""
    return compute_distance(
        X,
        Y,
        num_projections,
        kernel_fn,
        mus,
        num_mus,
        nu_shape,
        nu_ratio,
        normalise,
        p=p,
        metric="average",
    )


@partial(jit, static_argnames=["num_projections", "num_mus", "normalise", "nu_shape"])
def ekqd_centered(
    X,
    Y,
    num_projections,
    kernel_fn,
    mus=None,
    num_mus=None,
    nu_shape="flat",
    nu_ratio=1.0,
    normalise=True,
    p=2,
):
    """Compute the e-KQD-Centered distance."""
    return compute_distance(
        X,
        Y,
        num_projections,
        kernel_fn,
        mus,
        num_mus,
        nu_shape,
        nu_ratio,
        normalise,
        p=p,
        metric="centered",
    )


@partial(jit, static_argnames=["num_projections", "num_mus", "normalise", "nu_shape"])
def supkqd(
    X,
    Y,
    num_projections,
    kernel_fn,
    mus=None,
    num_mus=None,
    nu_shape="flat",
    nu_ratio=1.0,
    normalise=True,
    p=2,
):
    """Compute the sup-KQD distance."""
    return compute_distance(
        X,
        Y,
        num_projections,
        kernel_fn,
        mus,
        num_mus,
        nu_shape,
        nu_ratio,
        normalise,
        p=p,
        metric="max",
    )
