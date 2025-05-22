# source: https://bitbucket.org/TimotheeMathieu/monk-mmd/src/master/

import numpy as np


def blockMOM(K, x):
    """Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do blocks for.

    Returns
    -------

    list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
    """
    b = int(np.floor(len(x) / K))
    nb = K - (len(x) - b * K)
    nbpu = len(x) - b * K
    perm = np.random.permutation(len(x))
    blocks = [[(b + 1) * g + f for f in range(b + 1)] for g in range(nbpu)]
    blocks += [[nbpu * (b + 1) + b * g + f for f in range(b)] for g in range(nb)]
    return [perm[b] for b in blocks]


def MOM(x, blocks):
    """Compute the median of means of x using the blocks blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    """
    means_blocks = [np.mean([x[f] for f in ind]) for ind in blocks]
    indice = np.argsort(means_blocks)[int(np.floor(len(means_blocks) / 2)) + 1]
    return means_blocks[indice], indice


def mom(x, K):
    """Create K blocks randomly and compute the median of means of x using these blocks."""
    blocks = blockMOM(K, x)
    return MOM(x, blocks)[0]
