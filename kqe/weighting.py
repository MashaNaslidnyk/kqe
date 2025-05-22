from functools import partial

import jax.numpy as jnp
from jax import jit


def shape_slope_up(x, ratio):
    """
    For x in [0,1],
      y(0) = ratio (lowest),
      y(1) = 1 (highest).
    """
    return ratio + (1 - ratio) * x


def shape_slope_down(x, ratio):
    """
    For x in [0,1],
      y(0) = 1 (highest),
      y(1) = ratio (lowest).
    """
    return 1 - (1 - ratio) * x


def shape_triangle(x, ratio):
    """
    Edges = ratio, peak = 1 at x=0.5.
    Piecewise linear:
      left  (0 <= x <= 0.5),
      right (0.5 < x <= 1).
    """
    # left side: from ratio at x=0 up to 1 at x=0.5
    left_side = ratio + 2 * (1 - ratio) * x

    # right side: from 1 at x=0.5 down to ratio at x=1
    right_side = 1 + 2 * (ratio - 1) * (x - 0.5)
    return jnp.where(x <= 0.5, left_side, right_side)


def shape_reverse_triangle(x, ratio):
    """
    Edges = 1, trough = ratio at x=0.5.
    Piecewise linear:
      left  (0 <= x <= 0.5),
      right (0.5 < x <= 1).
    """
    # left side: from 1 at x=0 down to ratio at x=0.5
    left_side = 1 + 2 * (ratio - 1) * x

    # right side: from ratio at x=0.5 up to 1 at x=1
    right_side = ratio + 2 * (1 - ratio) * (x - 0.5)
    return jnp.where(x <= 0.5, left_side, right_side)


def shape_flat(x, ratio=None):
    """
    All ones => ratio is effectively 1 after normalization.
    """
    return jnp.ones_like(x)


@partial(jit, static_argnames=["num_points", "shape"])
def get_weights(num_points, shape="slope_up", ratio=0.5):
    """
    Returns an array of n weights that sum to 1,
    with 'lowest : highest = ratio : 1'.

    - ratio=0 => the lowest point(s) are 0, the highest is 1
    - ratio=1 => everything is the same (flat)
    - 0<ratio<1 => partial slope or partial difference
    """

    # Discretize [0,1] into n points
    x = jnp.linspace(0.0, 1.0, num_points)

    # Pick the shape function
    shape_functions = {
        "slope_up": shape_slope_up,
        "slope_down": shape_slope_down,
        "triangle": shape_triangle,
        "reverse_triangle": shape_reverse_triangle,
        "flat": shape_flat,
    }

    if shape not in shape_functions:
        raise ValueError(
            f"Unknown shape '{shape}'. Valid shapes: {list(shape_functions.keys())}"
        )

    # Compute unnormalized shape
    unnormalized = shape_functions[shape](x, ratio)

    # Clip negatives just in case, then normalize
    unnormalized = jnp.clip(unnormalized, 0, None)
    total = jnp.sum(unnormalized)

    # Use a small epsilon check in pure-JAX context (can't do 'if total < 1e-15:' outside of jit)
    def normalize_or_fallback(_vals, _sum):
        return jnp.where(_sum < 1e-15, jnp.ones_like(_vals) / _vals.size, _vals / _sum)

    w = normalize_or_fallback(unnormalized, total)
    return w


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("TkAgg")

    # Test
    ratio = 0.5
    npts = 100
    weights = get_weights(npts, shape="triangle", ratio=ratio)

    plt.figure(figsize=(6, 4))
    plt.plot(np.linspace(0, 1, npts), weights, marker="o", linestyle="-")
    plt.title(
        f"Triangle shape (n=5, {ratio=}. {weights.sum()=:.2f}. {weights[0]/weights[-1]:.2f})"
    )
    plt.xlabel("Index (1 to 5)")
    plt.ylabel("Weight")
    plt.ylim(0, 2 / npts)
    plt.grid(True)
    plt.show()
