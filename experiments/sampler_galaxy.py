# taken from https://github.com/antoninschrab/mmdfuse-paper/blob/master/sampler_galaxy.py
import jax
import jax.numpy as jnp
import torch
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
from jax import random


def load_images_list(highres):
    if highres:
        dataset = GalaxyMNISTHighrez(  # [3, 224, 224]
            root="./galaxy_data",
            download=True,
            train=False,
        )
    else:
        dataset = GalaxyMNIST(  # [3, 64, 64]
            root="./galaxy_data",
            download=True,
            train=False,
        )

    (custom_train_images, custom_train_labels), (
        custom_test_images,
        custom_test_labels,
    ) = dataset.load_custom_data(test_size=0.5, stratify=True)
    images = torch.cat((custom_train_images, custom_test_images))
    labels = torch.cat((custom_train_labels, custom_test_labels))

    images_list = (
        jnp.array(images[torch.where(labels == 3, True, False)]),
        jnp.array(images[torch.where(labels == 2, True, False)]),
        jnp.array(images[torch.where(labels == 1, True, False)]),
        jnp.array(images[torch.where(labels == 0, True, False)]),
    )

    return images_list


def sampler_galaxy(key, m, n, corruption, images_list):
    """
    For X: we sample uniformly from images with labels 3, 2, 1.
    For Y: with probability 'corruption' we sample uniformly from images with labels 3, 2, 1.
           with probability '1 - corruption' we sample uniformly from images with labels 0.
    """
    images_0, images_1, images_2, images_3 = images_list

    # X
    key, subkey = random.split(key)
    subkeys = random.split(subkey, num=5)
    choice = jax.random.choice(subkeys[0], jnp.arange(3), shape=(m,))
    m_0 = jnp.sum(choice == 0)  # m = m_0 + m_1 + m_2
    m_1 = jnp.sum(choice == 1)
    m_2 = jnp.sum(choice == 2)
    indices_0 = jax.random.permutation(
        subkeys[1],
        jnp.array(
            [
                False,
            ]
            * images_0.shape[0]
        )
        .at[:m_0]
        .set(True),
    )
    indices_1 = jax.random.permutation(
        subkeys[2],
        jnp.array(
            [
                False,
            ]
            * images_1.shape[0]
        )
        .at[:m_1]
        .set(True),
    )
    indices_2 = jax.random.permutation(
        subkeys[3],
        jnp.array(
            [
                False,
            ]
            * images_2.shape[0]
        )
        .at[:m_2]
        .set(True),
    )
    X = jnp.concatenate((images_0[indices_0], images_1[indices_1], images_2[indices_2]))
    X = jax.random.permutation(subkeys[4], X, axis=0)

    # Y
    key, subkey = random.split(key)
    subkeys = random.split(subkey, num=6)
    choice = jax.random.choice(
        subkeys[0],
        jnp.arange(4),
        shape=(n,),
        p=jnp.array(
            [
                (1 - corruption) / 3,
                (1 - corruption) / 3,
                (1 - corruption) / 3,
                corruption,
            ]
        ),
    )
    n_0 = jnp.sum(choice == 0)  # n = n_0 + n_1 + n_2 + n_3
    n_1 = jnp.sum(choice == 1)
    n_2 = jnp.sum(choice == 2)
    n_3 = jnp.sum(choice == 3)
    indices_0 = jax.random.permutation(
        subkeys[1],
        jnp.array(
            [
                False,
            ]
            * images_0.shape[0]
        )
        .at[:n_0]
        .set(True),
    )
    indices_1 = jax.random.permutation(
        subkeys[2],
        jnp.array(
            [
                False,
            ]
            * images_1.shape[0]
        )
        .at[:n_1]
        .set(True),
    )
    indices_2 = jax.random.permutation(
        subkeys[3],
        jnp.array(
            [
                False,
            ]
            * images_2.shape[0]
        )
        .at[:n_2]
        .set(True),
    )
    indices_3 = jax.random.permutation(
        subkeys[4],
        jnp.array(
            [
                False,
            ]
            * images_3.shape[0]
        )
        .at[:n_3]
        .set(True),
    )
    Y = jnp.concatenate(
        (
            images_0[indices_0],
            images_1[indices_1],
            images_2[indices_2],
            images_3[indices_3],
        )
    )
    Y = jax.random.permutation(subkeys[5], Y, axis=0)

    return X, Y
