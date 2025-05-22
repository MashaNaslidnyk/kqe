# testing and kernel stuff reimplemented without jax---it was tricky to get jax to work with mom

import json
from collections import OrderedDict
from dataclasses import dataclass
from os import path
from typing import Optional

import numpy as np
from mmd_mom import MMD_MOM
from tqdm.auto import tqdm

from kqe.local_config import DATADIR, LOGSDIR
from kqe.logging_utils import info, init_logging


def compute_median_heuristic(x1: np.ndarray, x2=None) -> float:
    """Compute the median heuristic for setting a Gaussian kernel's bandwidth."""
    if x2 is None:
        xs = x1
    else:
        xs = np.concatenate([x1, x2], axis=0)

    # Compute all pairwise squared distances divided by 2
    diffs = xs[:, None, :] - xs[None, :, :]
    sqdists = np.sum(diffs**2, axis=-1) / 2.0

    # Median distance and return its square root
    return float(np.sqrt(np.median(sqdists)))


def shuffle_cut_and_compute_distance(
    joint_samples: np.ndarray,
    permutation: np.ndarray,
    test_statistic,
    **test_statistic_kwargs,
) -> np.ndarray:
    """
    Shuffle the samples according to the permutation, split in half, and compute the test statistic.
    """
    n = joint_samples.shape[0] // 2
    permuted = joint_samples[permutation]
    left = permuted[:n]
    right = permuted[n:]
    return test_statistic(left, right, **test_statistic_kwargs)


def permutation_test(
    rng: np.random.RandomState,
    X: np.ndarray,
    Y: np.ndarray,
    test_statistic,
    num_permutations: int = 300,
    batch_size=None,
    level: float = 0.05,
    return_p_val: bool = False,
    **test_statistic_kwargs,
) -> np.ndarray:
    if batch_size is None:
        batch_size = num_permutations

    # observed statistic
    observed_stat = test_statistic(X, Y, **test_statistic_kwargs)

    # combine and prepare permutations
    joint = np.concatenate([X, Y], axis=0)
    n_total = joint.shape[0]

    # generate all permutations indices
    perms = np.array([rng.permutation(n_total) for _ in range(num_permutations)])

    # simulate in batches
    simulated = []
    for i in range(0, num_permutations, batch_size):
        batch = perms[i : i + batch_size]
        batch_stats = np.array(
            [
                shuffle_cut_and_compute_distance(
                    joint, p, test_statistic, **test_statistic_kwargs
                )
                for p in batch
            ]
        )
        simulated.append(batch_stats)

    simulated = np.concatenate(simulated)
    # include observed in null for threshold estimation
    all_stats = np.append(simulated, observed_stat)
    thresh = np.quantile(all_stats, 1 - level)

    if return_p_val:
        return np.mean(all_stats >= observed_stat)
    else:
        return np.array(observed_stat > thresh, dtype=int)


@dataclass(frozen=True)
class GaussianKernel:
    l: float = 1.0

    def __call__(self, x1: np.ndarray, x2: Optional[np.ndarray] = None) -> np.ndarray:
        """Gaussian (RBF) kernel:
        - If x1 and x2 are 1D, returns scalar k(x1,x2).
        - If x2 is None, returns the (n x n) Gram matrix for x1.
        - Otherwise, returns the (n x m) Gram matrix between rows of x1 and x2.
        """
        # Ensure inputs are at least 2D arrays
        X = np.atleast_2d(x1)
        if x2 is None:
            Y = X
        else:
            Y = np.atleast_2d(x2)

        # Compute squared norms
        X_norms = np.sum(X**2, axis=1)[:, np.newaxis]  # shape (n,1)
        Y_norms = np.sum(Y**2, axis=1)[np.newaxis, :]  # shape (1,m)

        # Squared Euclidean distances
        sqdist = X_norms + Y_norms - 2 * np.dot(X, Y.T)
        # Apply kernel
        K = np.exp(-0.5 * sqdist / (self.l**2))

        # If originally passed two 1D vectors, return scalar
        if (
            x1.ndim == 1
            and x2 is not None
            and isinstance(x2, np.ndarray)
            and x2.ndim == 1
        ):
            return float(K[0, 0])
        return K


def mom(X, Y, kernel_fn):
    Q = int(X.shape[0] / np.log(X.shape[0]))
    test = MMD_MOM(Q=Q, kernel=kernel_fn, solver="BCD_Fast")
    res = test.estimate(X, Y)
    return res


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Define experiment and initialize logging
    # ---------------------------------------------------------------------
    experiment_name = "power_decay_w_mom"
    init_logging(filename=path.join(LOGSDIR, f"{experiment_name}.log"))

    # ---------------------------------------------------------------------
    # Define parameters
    # -------------------------------------------------------------------
    dims = [32, 64, 128, 256, 512]
    num_samples = 200
    num_runs = 300

    # -------------------------------------------------------------------------
    # Define distances
    # -------------------------------------------------------------------------
    distances = OrderedDict([("mom", mom)])
    rejection_rate_dict = OrderedDict([("mom", {dim: -1 for dim in dims})])
    distance_fn_kwargs = {}  # define if needed

    info("Starting experiments...")
    for dim in dims:
        for distance_name, distance_fn in tqdm(distances.items(), desc=f"{dim=}"):
            rejections = []

            for seed in tqdm(range(num_runs), desc=f"{distance_name}"):
                rng = np.random.RandomState(seed)

                diag = np.ones(dim)
                diag[:3] = 4
                Sigma = np.diag(diag)
                X = rng.multivariate_normal(
                    mean=np.zeros(dim), cov=np.eye(dim), size=100
                )
                Y = rng.multivariate_normal(mean=np.zeros(dim), cov=Sigma, size=100)

                kernel_fn = GaussianKernel(l=float(compute_median_heuristic(X, Y)))

                result = permutation_test(
                    rng,
                    X,
                    Y,
                    test_statistic=distance_fn,
                    num_permutations=300,
                    level=0.05,
                    return_p_val=False,
                    **dict(
                        kernel_fn=kernel_fn, **distance_fn_kwargs.get(distance_name, {})
                    ),
                )
                rejections.append(result)

            rejection_rate = float(np.mean(np.array(rejections)))
            rejection_rate_dict[distance_name][dim] = rejection_rate

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
