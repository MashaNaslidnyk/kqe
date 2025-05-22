from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array, jit, random, vmap

from kqe.logging_utils import info

TestStatisticLike = Callable[[Array, Array, ...], Array]


def shuffle_cut_and_compute_distance(
    joint_samples: Array,
    permutation: Array,
    test_statistic: TestStatisticLike,
    **test_statistic_kwargs,
) -> Array:
    """Shuffle the samples according to the permutation, cut them in half, and compute the distance between the two
    halves."""
    num_samples = int(joint_samples.shape[0] / 2)
    left_samples, right_samples = (
        joint_samples[permutation][:num_samples],
        joint_samples[permutation][num_samples:],
    )

    res = test_statistic(left_samples, right_samples, **test_statistic_kwargs)

    return res


@partial(
    jit,
    static_argnames=[
        "test_statistic",
        "num_permutations",
        "batch_size",
        "return_p_val",
        # all the below may be in test_statistic_kwargs---depending on the test_statistic
        "num_diagonals",  # used by mmd_multi_diagonal_estimator
        "num_projections",  # used by all KQD distances
        "num_mus",  # used by all KQD distances
        "normalise",  # used by all KQD distances
        "nu_shape",  # used by all KQD distances
        "num_nystrom_points",  # used by nystrom_mmd_sq
        "num_random_me_points",  # used by me_test_statistic
        "key_test",  # used by nystrom_mmd_sq, me_test_statistic
    ],
)
def permutation_test(
    key_perm: random.PRNGKey,
    X: Array,
    Y: Array,
    test_statistic: TestStatisticLike,
    num_permutations: int = 300,
    batch_size: Optional[int] = None,
    level: float = 0.05,
    return_p_val: bool = False,
    **test_statistic_kwargs,
) -> Array:
    """
    Perform a permutation test with batching to handle large memory requirements.

    :param key_perm: random key to use for generating permutations
    :param X: samples from the first distribution
    :param Y: samples from the second distribution
    :param num_permutations: Total number of permutations to use for the permutation test
    :param level: Critical value for the test
    :param batch_size: Number of permutations to process per batch
    :param return_p_val: Whether to return the p-value or a binary decision
    :param test_statistic_kwargs: Extra parameters for the test statistic

    :return: p-value or binary decision
    """
    info(
        f"Compiled permutation_test with {test_statistic=}, {num_permutations=}, {batch_size=}, {return_p_val=}, {test_statistic_kwargs=}"
    )

    if batch_size is None:
        batch_size = num_permutations

    observed_distances = test_statistic(X, Y, **test_statistic_kwargs)

    # Simulate the null distribution
    joint_samples = jnp.concatenate([X, Y])

    # Generate all permutations at once
    all_permutations = vmap(
        lambda key: random.permutation(key, joint_samples.shape[0])
    )(random.split(key_perm, num_permutations))

    # Process permutations in batches
    def compute_batch_distances(batch_permutations):
        return vmap(
            lambda permutation: shuffle_cut_and_compute_distance(
                joint_samples, permutation, test_statistic, **test_statistic_kwargs
            )
        )(batch_permutations)

    num_batches = (num_permutations + batch_size - 1) // batch_size
    simulated_distances = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_permutations)
        batch_permutations = all_permutations[start_idx:end_idx]
        simulated_distances.append(compute_batch_distances(batch_permutations))

    # Concatenate results and compute critical value
    simulated_distances = jnp.concatenate(simulated_distances)
    simulated_distances = jnp.append(simulated_distances, observed_distances)

    critical_value = jnp.quantile(simulated_distances, 1 - level)

    if return_p_val:
        p_val = (simulated_distances >= observed_distances).mean()
        return p_val
    else:
        return (observed_distances > critical_value).astype(jnp.int32)
