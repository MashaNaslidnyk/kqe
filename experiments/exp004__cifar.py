import json
from collections import OrderedDict
from math import prod
from os import path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision.transforms as transforms
from jax import random
from torchvision import datasets
from tqdm.auto import tqdm  # or just 'from tqdm import tqdm', if you prefer

from kqe.kernels import GaussianKernel, compute_median_heuristic
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

    experiment_name = "cifar"
    init_logging(filename=path.join(LOGSDIR, f"{experiment_name}.log"))

    # -------------------------------------------------------------------------
    # Define parameters
    # -------------------------------------------------------------------------
    num_samples_lst = [100, 500, 1000, 1500, 2000]
    img_size = 64
    batch_size = 20
    num_runs = 200

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    dataset_test = datasets.CIFAR10(
        root="../data/cifar_data/cifar10",
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=10000,
        shuffle=False,  # Disable shuffling
        num_workers=1,
    )

    # Obtain CIFAR10 images
    for i, (imgs, _) in enumerate(dataloader_test):  # Ignore labels
        cifar10 = imgs.numpy()
        break  # We only need one batch here

    # Shuffle the data manually using JAX
    key = random.PRNGKey(123)
    key, key_shuffle = random.split(key, 2)
    indices = random.permutation(key_shuffle, cifar10.shape[0])
    cifar10 = cifar10[indices]

    # Obtain CIFAR10.1 images
    data_new = np.load("../data/cifar_data/cifar10.1_v4_data.npy")
    data_T = np.transpose(data_new, [0, 3, 1, 2])
    TT = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trans = transforms.ToPILImage()
    cifar10p1 = torch.zeros([len(data_T), 3, img_size, img_size])
    data_T_tensor = torch.from_numpy(data_T)
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        cifar10p1[i] = TT(d0)

    cifar10p1 = cifar10p1.numpy()

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
            ("ekqd_tr", ekqd),
            ("ekqd_rtr", ekqd),
            ("ekqd_sup", ekqd),
            ("ekqd_sdown", ekqd),
            ("ekqd_2_mu_normal", ekqd),
            ("ekqd_2_mu_uniform", ekqd),
        ]
    )

    # Prepare a dict to store rejection rates
    rejection_rate_dict = {
        distance_name: {num_samples: -1 for num_samples in num_samples_lst}
        for distance_name in distances.keys()
    }

    # -------------------------------------------------------------------------
    # Main loop over sample sizes and distances
    # -------------------------------------------------------------------------

    info("Starting experiments...")

    for num_samples in num_samples_lst:
        # Distances may have parameter settings depending on num_samples
        num_mus = int(jnp.log(num_samples))

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
            f"ekqd_2_mu_normal": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "p": 2,
            },
            f"ekqd_2_mu_uniform": {
                "num_projections": int(jnp.log(num_samples)),
                "normalise": True,
                "p": 2,
            },
        }

        for distance_name, distance_fn in tqdm(
            distances.items(), desc=f"{num_samples=}"
        ):
            # Clear JAX caches before each run (optional, but can help)
            jax.clear_caches()

            rejections = []

            # Need to start with the same key for each distance for reproducibility, otherwise the ordering of distances changes the outcome
            key_runs = key
            for _ in tqdm(range(num_runs), desc=f"{distance_name}"):
                key_runs, key_X, key_Y, key_perm = random.split(key_runs, 4)

                # Collect CIFAR10 images
                ind_cifar10 = random.choice(
                    key_X, len(cifar10), (num_samples,), replace=False
                )
                X = cifar10[ind_cifar10.tolist()]

                # Collect CIFAR10.1 images
                ind_cifar10p1 = random.choice(
                    key_Y, len(cifar10p1), (num_samples,), replace=False
                )
                Y = cifar10p1[ind_cifar10p1.tolist()]

                # Reshape to 2D (samples x features) and convert to numpy
                X = X.reshape(X.shape[0], -1)
                Y = Y.reshape(Y.shape[0], -1)

                # Create the kernel
                kernel_fn = GaussianKernel(l=float(compute_median_heuristic(X, Y)))

                # Set mu for experiments that do non-default mus (gaussian and uniform). Cos they depend on X, must do them here
                # keep key_runs unaffected, for reproducibility
                if distance_name.endswith("mu_normal"):
                    q25, q75 = jnp.percentile(
                        jnp.vstack([X, Y]), jnp.array([25, 75]), axis=0
                    )
                    iqr = q75 - q25
                    distance_fn_kwargs[distance_name]["mus"] = (
                        iqr / 1.349
                    ) * random.normal(key_runs, (num_mus,) + X.shape[1:])
                elif distance_name.endswith("mu_uniform"):
                    q25, q75 = jnp.percentile(
                        jnp.vstack([X, Y]), jnp.array([25, 75]), axis=0
                    )
                    iqr = q75 - q25
                    distance_fn_kwargs[distance_name]["mus"] = iqr * (
                        2
                        * random.uniform(
                            key_runs, (num_mus,) + (prod(cifar10.shape[1:]),)
                        )
                        - 1
                    )

                # Run the test
                result = permutation_test(
                    key_perm=key_perm,
                    X=X,
                    Y=Y,
                    test_statistic=distance_fn,
                    num_permutations=300,
                    batch_size=50,
                    level=0.05,
                    return_p_val=False,
                    **dict(
                        kernel_fn=kernel_fn, **distance_fn_kwargs.get(distance_name, {})
                    ),
                )
                rejections.append(result)

            # Save the average rejection rate
            rejection_rate = float(jnp.mean(jnp.array(rejections)))
            rejection_rate_dict[distance_name][num_samples] = rejection_rate

            distance_fn_kwargs["ekqd_2_normal"]["mus"] = None
            distance_fn_kwargs["ekqd_2_uniform"]["mus"] = None

            # Write partial results to JSON
            json.dump(
                {
                    "rejection_rate": rejection_rate_dict,
                    "num_samples_lst": num_samples_lst,
                    "distances": list(distances.keys()),
                    "img_size": img_size,
                    "batch_size": batch_size,
                    "num_runs": num_runs,
                    "experiment_name": experiment_name,
                    "distance_fn_kwargs": distance_fn_kwargs,
                },
                open(path.join(DATADIR, f"{experiment_name}.json"), "w"),
            )
            info(f"{distance_name}, {num_samples=}: {rejection_rate}")

    info("Done.")
