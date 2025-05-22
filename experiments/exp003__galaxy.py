import json
from collections import OrderedDict
from os import path

import jax.numpy as jnp
from jax import clear_caches, random
from sampler_galaxy import load_images_list, sampler_galaxy
from tqdm.auto import tqdm

from kqe.kernels import GaussianKernel, compute_median_heuristic
from kqe.kqd import ekqd, ekqd_centered, supkqd
from kqe.local_config import DATADIR, LOGSDIR
from kqe.logging_utils import info, init_logging
from kqe.mmd import (me_test_statistic, mmd_linear_estimator,
                     mmd_multi_diagonal_estimator, mmd_squared_U_statistic,
                     nystrom_mmd_sq)
from kqe.testing import permutation_test

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Define experiment and initialize logging
    # ---------------------------------------------------------------------
    experiment_name = "galaxy"
    init_logging(filename=path.join(LOGSDIR, f"{experiment_name}.log"))

    # ---------------------------------------------------------------------
    # Define parameters
    # ---------------------------------------------------------------------
    num_runs = 200
    corruption = 0.15
    num_samples_lst = [100, 500, 1000, 1500, 2000, 2500]

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    images_list = load_images_list(highres=False)

    # ---------------------------------------------------------------------
    # Define distances
    # ---------------------------------------------------------------------
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
            ("ekqd_tr", ekqd),
            ("ekqd_rtr", ekqd),
            ("ekqd_sup", ekqd),
            ("ekqd_sdown", ekqd),
            ("nystrom", nystrom_mmd_sq),
            ("me", me_test_statistic),
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
        distance_fn_kwargs = {
            f"ekqd_1": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 1,
            },
            f"ekqd_2": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 2,
            },
            f"ekqd_centered_1": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 1,
            },
            f"ekqd_centered_2": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 2,
            },
            f"supkqd_1": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 1,
            },
            f"supkqd_2": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "p": 2,
            },
            f"ekqd_tr": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "nu_shape": "triangle",
                "nu_ratio": 0.0,
                "p": 2,
            },
            f"ekqd_rtr": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "nu_shape": "reverse_triangle",
                "nu_ratio": 0.0,
                "p": 2,
            },
            f"ekqd_sup": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "nu_shape": "slope_up",
                "nu_ratio": 0.0,
                "p": 2,
            },
            f"ekqd_sdown": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "num_mus": int(jnp.log(num_samples)),
                "nu_shape": "slope_down",
                "nu_ratio": 0.0,
                "p": 2,
            },
            f"mmdmulti": {"num_diagonals": int(jnp.log(num_samples)) ** 2},
            "nystrom": {
                "num_nystrom_points": int(jnp.log(num_samples)) ** 2,
                "key_test": None,
            },
            "me": {
                "num_random_me_points": int(jnp.log(num_samples)) ** 2,
                "key_test": None,
            },
        }
        for distance_name, distance_fn in tqdm(
            distances.items(), desc=f"{num_samples=}"
        ):
            clear_caches()

            rejections = []

            # Need to start with the same key for each distance for reproducibility, otherwise the ordering of distances changes the outcome
            key_runs = key
            for i in tqdm(range(num_runs), desc=f"{distance_name}"):
                key_runs, key_test, subkey = random.split(key_runs, 3)

                # Slightly ugly special casing for the two test statistics that require a key to work
                distance_fn_kwargs.get("nystrom", {"key_test": None})[
                    "key_test"
                ] = key_test
                distance_fn_kwargs.get("me", {"key_test": None})["key_test"] = key_test

                X, Y = sampler_galaxy(
                    subkey,
                    m=num_samples,
                    n=num_samples,
                    corruption=corruption,
                    images_list=images_list,
                )
                X = jnp.array(X, dtype=jnp.float32).reshape((X.shape[0], -1))
                Y = jnp.array(Y, dtype=jnp.float32).reshape((Y.shape[0], -1))

                kernel_fn = GaussianKernel(l=float(compute_median_heuristic(X, Y)))

                key, subkey = random.split(key)

                result = permutation_test(
                    key_perm=subkey,
                    X=X,
                    Y=Y,
                    test_statistic=distance_fn,
                    num_permutations=200,
                    batch_size=100,
                    level=0.05,
                    return_p_val=False,
                    **dict(
                        kernel_fn=kernel_fn, **distance_fn_kwargs.get(distance_name, {})
                    ),
                )
                rejections.append(result)

            rejection_rate = float(jnp.mean(jnp.array(rejections)))
            rejection_rate_dict[distance_name][num_samples] = rejection_rate

            # cos the key isn't json serialisable
            distance_fn_kwargs.get("nystrom", {"key_test": None})["key_test"] = None
            distance_fn_kwargs.get("me", {"key_test": None})["key_test"] = None

            json.dump(
                {
                    "rejection_rate": rejection_rate_dict,
                    "num_samples_lst": num_samples_lst,
                    "corruption": corruption,
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
