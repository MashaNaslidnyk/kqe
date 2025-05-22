import json
from collections import OrderedDict
from os import path

import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from kqe.kernels import (GaussianKernel, PolynomialKernel,
                         compute_median_heuristic)
from kqe.kqd import ekqd, ekqd_centered, supkqd
from kqe.local_config import DATADIR, LOGSDIR
from kqe.logging_utils import info, init_logging
from kqe.mmd import (mmd_linear_estimator, mmd_multi_diagonal_estimator,
                     mmd_squared_U_statistic)
from kqe.testing import permutation_test

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Define experiment and initialize logging
    # ---------------------------------------------------------------------
    experiment_name = "power_decay"
    init_logging(filename=path.join(LOGSDIR, f"{experiment_name}.log"))

    # ---------------------------------------------------------------------
    # Define parameters
    # ---------------------------------------------------------------------
    dims = [32, 64, 128, 256, 512]
    num_samples = 200
    num_runs = 300

    # -------------------------------------------------------------------------
    # Define distances
    # -------------------------------------------------------------------------
    distances = OrderedDict(
        [
            ("ekqd_1", ekqd),
            ("ekqd_2", ekqd),
            ("ekqd_centered_1", ekqd_centered),
            ("ekqd_centered_2", ekqd_centered),
            ("supkqd_1", supkqd),
            ("supkqd_2", supkqd),
            ("mmdmulti", mmd_multi_diagonal_estimator),
            ("mmdlin", mmd_linear_estimator),
            ("mmd", mmd_squared_U_statistic),
            ("esw", ekqd),
            ("esw_mu_normal", ekqd),
        ]
    )

    rejection_rate_dict = OrderedDict(
        [
            (distance_name, {dim: -1 for dim in dims})
            for distance_name in distances.keys()
        ]
    )

    distance_fn_kwargs = {
        "ekqd_1": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 1,
        },
        "ekqd_2": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 2,
        },
        "ekqd_centered_1": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 1,
        },
        "ekqd_centered_2": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 2,
        },
        "supkqd_1": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 1,
        },
        "supkqd_2": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 2,
        },
        "mmdmulti": {"num_diagonals": int(jnp.log(num_samples)) ** 2},
        "esw": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 2,
        },
        "esw_mu_normal": {
            "num_projections": int(jnp.log(num_samples)),
            "normalise": True,
            "num_mus": int(jnp.log(num_samples)),
            "p": 2,
        },
    }

    key = random.PRNGKey(123)

    # -------------------------------------------------------------------------
    # Main loop over dimensionalities and distances
    # -------------------------------------------------------------------------

    info("Starting experiments...")
    num_mus = int(jnp.log(num_samples))

    for dim in dims:
        diag_elements = jnp.ones(dim)
        diag_elements = diag_elements.at[:3].set(4)  # Scale first three entries
        Sigma = jnp.diag(diag_elements)

        for distance_name, distance_fn in tqdm(distances.items(), desc=f"{dim=}"):
            rejections = []

            key_runs = key

            for _ in tqdm(range(num_runs), desc=f"{distance_name}"):
                key_runs, key_perm, key_X, key_Y = random.split(key_runs, 4)

                # Generate data
                X = jax.random.multivariate_normal(
                    key_X, mean=jnp.zeros(dim), cov=jnp.eye(dim), shape=(num_samples,)
                )
                Y = jax.random.multivariate_normal(
                    key_Y, mean=jnp.zeros(dim), cov=Sigma, shape=(num_samples,)
                )

                # Kernel with median heuristic
                if distance_name.startswith("esw"):
                    kernel_fn = PolynomialKernel(c=0, d=1)
                else:
                    kernel_fn = GaussianKernel(l=float(compute_median_heuristic(X, Y)))

                if distance_name.endswith("mu_normal"):
                    q25, q75 = jnp.percentile(
                        jnp.vstack([X, Y]), jnp.array([25, 75]), axis=0
                    )
                    iqr = q75 - q25
                    distance_fn_kwargs[distance_name]["mus"] = (
                        iqr / 1.349
                    ) * random.normal(key_runs, (num_mus,) + X.shape[1:])

                # Permutation test
                result = permutation_test(
                    key_perm=key_perm,
                    X=X,
                    Y=Y,
                    test_statistic=distance_fn,
                    num_permutations=300,
                    batch_size=150,
                    level=0.05,
                    **dict(
                        kernel_fn=kernel_fn, **distance_fn_kwargs.get(distance_name, {})
                    ),
                )
                rejections.append(result)

            rejection_rate = float(jnp.mean(jnp.array(rejections)))
            rejection_rate_dict[distance_name][dim] = rejection_rate

            if distance_name.endswith("mu_normal"):
                # cos it's not json serializable
                distance_fn_kwargs[distance_name]["mus"] = None

            json.dump(
                {
                    "rejection_rate": rejection_rate_dict,
                    "dims": dims,
                    "distances": list(distances.keys()),
                    "num_runs": num_runs,
                    "experiment_name": experiment_name,
                    "distance_fn_kwargs": distance_fn_kwargs,
                },
                open(path.join(DATADIR, f"{experiment_name}.json"), "w"),
            )

            info(
                f"Rejection rate for {distance_name}, {dim=} : {rejection_rate_dict[distance_name][dim]}"
            )

info("Done.")
