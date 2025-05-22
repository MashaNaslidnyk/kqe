import json
from collections import OrderedDict
from os import path

import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from kqe.kernels import PolynomialKernel
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
    experiment_name = "laplace_vs_gaussian"
    init_logging(filename=path.join(LOGSDIR, f"{experiment_name}.log"))

    # ---------------------------------------------------------------------
    # Define parameters
    # ---------------------------------------------------------------------
    num_samples_lst = [100, 500, 2000, 5000, 10000]
    dim = 1
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
        ]
    )

    rejection_rate_dict = OrderedDict(
        [
            (distance_name, {num_samples: -1 for num_samples in num_samples_lst})
            for distance_name in distances.keys()
        ]
    )

    key = random.PRNGKey(123)

    # ---------------------------------------------------------------------
    # 5. Main loop over sample sizes and distances
    # ---------------------------------------------------------------------

    info("Starting experiments...")

    for num_samples in num_samples_lst:
        data_shape = (num_samples, dim)

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
        }

        for distance_name, distance_fn in tqdm(
            distances.items(), desc=f"{num_samples=}"
        ):
            rejections = []

            key_runs = key

            for _ in tqdm(range(num_runs), desc=f"{distance_name}"):
                key_runs, key_perm, key_sigma, key_laplace, key_normal = random.split(
                    key_runs, 5
                )

                sigma = random.uniform(
                    key=key_sigma, minval=0.5, maxval=1.0
                )  # variance of Gaussian

                mean = 0.0  # Gaussian mean
                variance = sigma**2  # Gaussian variance
                scale = jnp.sqrt(variance / 2)  # Laplace scale parameter

                X = random.laplace(key_laplace, shape=data_shape) * scale + mean
                std = jnp.sqrt(variance)  # Gaussian std

                Y = random.normal(key_normal, shape=data_shape) * std + mean

                kernel_fn = PolynomialKernel(c=1, d=3)

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
            rejection_rate_dict[distance_name][num_samples] = rejection_rate

            json.dump(
                {
                    "rejection_rate": rejection_rate_dict,
                    "num_samples_lst": num_samples_lst,
                    "distances": list(distances.keys()),
                    "num_runs": num_runs,
                    "experiment_name": experiment_name,
                    "distance_fn_kwargs": distance_fn_kwargs,
                },
                open(path.join(DATADIR, f"{experiment_name}.json"), "w"),
            )

            info(
                f"Rejection rate for {distance_name}, {num_samples=} : {rejection_rate_dict[distance_name][num_samples]}"
            )

info("Done.")
