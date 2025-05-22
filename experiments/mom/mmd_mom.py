# source: https://bitbucket.org/TimotheeMathieu/monk-mmd/src/master/

import inspect
import time
import warnings

import numpy as np
from mom_estimator import blockMOM


class MMD_MOM:
    """Estimate the MMD (maximum mean discrepancy) using MOM (median of means) estimate. The result is robust to outliers and faster in some case (particularly when solver is "MOM_Ite_Cost"

    Parameters
    ----------

    Q : int, odd
        number of blocks to use for MOM. Usually, the larger Q is, the faster the algorithm will be and the more robust it will be.
        Conversely, if Q is large, the efficiency of the estimation will suffer.

    solver : string in {'BCD','BCD_Fast','QCQP'}
        solver to use to estimator MMD.

    kernel_type in {'function','graam_matrix'}

    kernel :  default None
        kernel function or Graam matrix. If None, then use the function np.exp(-np.linalg.norm(x-y)^2*nb_features). If it is a function it must be such that kernel(X) return the Graam matrix of X and kernel(X,Y) return the matrix of kernel (k(x_i,y_j))_{i,j}.

    maxiter : int, default = 100
        number of iterations, used only for solvers in {'SGD','SGD_fixed_blocks','Projected_Gradient'}

    epsilon : float, default=1e-5
        coefficient of the L2 regularisation used in order to make the Gram matrix inversible

    perte : boolean, default = False
        wether or not to save the mmd at each iteration in the list self.pertes

    beta : float, default=1e-3
        parameter for the step size for gradient method.

    nb_shuffle : int, default=5
        number of shuffle used in BCD Fast

    stall : int, default=10
        Wait stall steps before doing the shuffles in the BCD-Fast algorithm

    Returns
    -------

    float, estimate of MMD.

    """

    def __init__(
        self,
        Q=11,
        solver="BCD",
        kernel_type="function",
        kernel=None,
        tol=0.01,
        maxiter=100,
        epsilon=1e-5,
        perte=False,
        beta=0.001,
        nb_shuffle=5,
        stall=10,
    ):

        if Q % 2 == 0:
            raise ValueError("Q is even")
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def estimate(self, y1, y2):
        if self.solver == "BCD":
            return self._estimate_bcd(y1, y2)
        elif self.solver == "BCD_Fast":
            return self._estimate_bcd_fast(y1, y2)
        else:
            raise ValueError("There is no solver with that name implemented.")

    def _compute_matrices(self, y1, y2, blocks, fast):
        # Computation of the Graam matrice(s) and Cholesky decomposition(s)
        kernel = self.kernel
        if self.kernel_type == "function":
            if fast:
                G = [
                    kernel(np.vstack((y1[blocks[q]], y2[blocks[q]])))
                    + self.epsilon * np.eye(2 * len(blocks[q]))
                    for q in range(self.Q)
                ]
                l = [np.linalg.cholesky(G[q]).T for q in range(self.Q)]

            else:
                G = kernel(np.vstack((y1, y2))) + self.epsilon * np.eye(2 * len(y1))
                l = np.linalg.cholesky(G).T

        elif self.kernel_type == "matrix":
            if fast:
                warnings.warn(
                    "If you have access to the function, it may be faster to put a function kernel_type because then the whole kernel function would not be computed but only the kernel function on blocks."
                )
                ind = [
                    np.hstack([blocks[q], blocks[q] + len(y1)]) for q in range(self.Q)
                ]
                G = [
                    kernel[ind[q]][ind[q]] + self.epsilon * np.eye(2 * len(blocks[q]))
                    for q in range(self.Q)
                ]
                l = [np.linalg.cholesky(G[q]).T for q in range(self.Q)]
            else:
                G = kernel
                l = np.linalg.cholesky(kernel).T
        return G, l

    def _estimate_bcd(self, y1, y2):

        # Verify that len(y1)=len(y2)
        if (y1 is not None) and (y2 is not None):
            if len(y1) != len(y2):
                raise ValueError("len(y1) is not equal to len(y2)")
        G, l = self._compute_matrices(y1, y2, None, False)
        if y1 is None:
            y1 = np.zeros(len(G))
        c = np.ones(2 * len(y1))
        c = c / np.sqrt(c.dot(G).dot(c))

        a = time.time()
        res = 0
        self.pertes = []
        for t in range(self.maxiter):
            pas = 1 / (1 + self.beta * t)

            # Sample the blocks
            blocks = blockMOM(self.Q, y1)

            # Compute the block which realize the median of the cost
            perte = [
                np.sum(G[blocks[q]] / len(blocks[q]), axis=0).dot(c)
                - np.sum(G[blocks[q] + len(y1)] / len(blocks[q]), axis=0).dot(c)
                for q in range(self.Q)
            ]
            if self.perte:
                self.pertes += [np.median(perte)]
            bmed = np.argsort(np.array(perte).ravel())[int(np.floor(self.Q / 2))]

            # do the maximization on the median block
            c_tilde = np.zeros(2 * len(y1))
            c_tilde[blocks[bmed]] = 1
            c_tilde[blocks[bmed] + len(y1)] = -1
            c_tilde = c_tilde / np.linalg.norm(
                l[:, np.hstack([blocks[bmed], blocks[bmed] + len(y1)])].dot(
                    np.hstack([np.ones(len(blocks[bmed])), -np.ones(len(blocks[bmed]))])
                )
            )
            c = c + pas * (c_tilde - c)

        return perte[bmed]

    def _estimate_bcd_fast(self, y1, y2):
        # Verify that len(y1)=len(y2)
        if (y1 is not None) and (y2 is not None):
            if len(y1) != len(y2):
                raise ValueError("len(y1) is not equal to len(y2)")
        if y1 is None:
            y1 = np.zeros(len(y2))
        nb_shuffle = self.nb_shuffle - 1

        blocks = blockMOM(self.Q, y1)
        G, l = self._compute_matrices(y1, y2, blocks, True)
        c = [np.ones(2 * len(blocks[q])) for q in range(self.Q)]
        c = [c[q] / np.sqrt(c[q].dot(G[q]).dot(c[q])) for q in range(self.Q)]
        lengths = [len(blocks[q]) for q in range(self.Q)]
        self.pertes = []
        for t in range(self.maxiter):
            pas = 1 / (1 + self.beta * t)

            if (
                (t > self.stall)
                and (nb_shuffle != 0)
                and (
                    (t - self.stall)
                    % np.floor((self.maxiter - self.stall) / nb_shuffle)
                    == 0
                )
            ):
                blocks = blockMOM(self.Q, y1)
                G, l = self._compute_matrices(y1, y2, blocks, True)

            # Compute the block which realize the median of the cost
            perte = [
                np.sum(G[q][: lengths[q]] / lengths[q], axis=0).dot(c[q])
                - np.sum(G[q][lengths[q] :] / lengths[q], axis=0).dot(c[q])
                for q in range(self.Q)
            ]

            if self.perte:
                self.pertes += [np.median(perte)]
            bmed = np.argsort(np.array(perte).ravel())[int(np.floor(self.Q / 2))]

            # do the maximization on the median block
            c_tilde = l[bmed].dot(
                np.hstack([np.ones(len(blocks[bmed])), -np.ones(len(blocks[bmed]))])
            )

            norm = np.linalg.norm(c_tilde)
            c[bmed] = c[bmed] + pas * (
                np.hstack([np.ones(len(blocks[bmed])), -np.ones(len(blocks[bmed]))])
                / norm
                - c[bmed]
            )

        return perte[bmed]
