from abc import ABC, abstractmethod
from typing import Callable, Optional

import jax.numpy as jnp
from chex import dataclass
from jax import Array, jit, vmap

KernelLike = Callable[[Array, Array], Array]


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array) -> Array:
        pass


@dataclass(frozen=True, eq=True)
class GaussianKernel(Kernel):
    l: Optional[float] = 1.0

    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / self.l**2)


@dataclass(frozen=True, eq=True)
class PolynomialKernel(Kernel):
    c: Optional[float] = 1.0
    d: Optional[int] = 3

    def __call__(self, x1: Array, x2: Array) -> Array:
        return (jnp.dot(x1, x2) + self.c) ** self.d


@dataclass(frozen=True, eq=True)
class PolynomialNormalisedKernel(Kernel):
    c: Optional[float] = 1.0
    d: Optional[int] = 3

    def __call__(self, x1: Array, x2: Array) -> Array:
        return (jnp.dot(x1, x2) + self.c) ** self.d / (
            (jnp.dot(x1, x1) + self.c) ** (self.d / 2)
            * (jnp.dot(x2, x2) + self.c) ** (self.d / 2)
        )


@jit
def compute_median_heuristic(x1: Array, x2: Optional[Array] = None) -> jnp.ndarray:
    """Compute the median heuristic."""
    if x2 is None:
        xs = x1
    else:
        xs = jnp.concatenate([x1, x2], axis=0)

    distances = vmap(lambda xa: vmap(lambda xb: ((xa - xb) ** 2).sum() / 2)(xs))(xs)
    return jnp.sqrt(jnp.median(distances))
