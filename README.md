# KQE: Kernel Quantile Embeddings and Discrepancies

A Python library for computing Kernel Quantile Discrepancies (KQDs) based on Kernel Quantile Embeddings (KQEs), proposed in PAPER.

## Features

* **Main module**: `kqd`
* **Backends**: Built on [JAX](https://github.com/google/jax) for accelerated, differentiable computations.
* **Experiments**: Experiments from the PAPER are implemented in the `experiments/` folder.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mashanaslidnyk/kqe.git
cd kqe
```

### 2. Install the package

* **Editable mode (recommended for development):**

  ```bash
  make dev-install
  ```
* **Editable mode that also allows you run experiments:**

  To install in editable mode, and also install dependencies required to run the experiments in the `experiments/` folder:

  ```bash
  make dev-install-experiments
  ```
* **Standard install:**

  ```bash
  make install
  ```

## Quick Start

```python
import jax.numpy as jnp
from kqe.kernels import GaussianKernel
from kqe.kqd import ekqd

# Sample data
X = jnp.array([[1.0], [2.0], [3.0]])
Y = jnp.array([[1.5], [2.5], [3.5]])

# Kernel
k = GaussianKernel(l=1.0)

# Compute e-KQD²
ekqd_val = ekqd(X, Y, kernel_fn=k, num_projections=3)
print("eKQD²:", ekqd_val)
```

## Testing & Formatting

To run pytest tests:

```bash
make test
```

To format everything with black and isort:

```bash
make format
```

## License

This project is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details.
