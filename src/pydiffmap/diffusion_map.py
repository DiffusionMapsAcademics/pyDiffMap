# -*- coding: utf-8 -*-
"""
Routines and Class definitions for the diffusion maps algorithm.
"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from . import kernel


class DiffusionMap(object):
    """
    Diffusion Map object to be used in data analysis for fun and profit.

    Parameters
    ----------
    alpha : scalar, optional
        Exponent to be used for the left normalization in constructing the diffusion map.
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    kernel_type : string, optional
        Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
    epsilon: string or scalar, optional
        Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) or 'bgh' (Berry, Giannakis and Harlim).
    n_evecs : int, optional
        Number of diffusion map eigenvectors to return
    neighbor_params : dict or None, optional
        Optional parameters for the nearest Neighbor search. See scikit-learn NearestNeighbors class for details.
    metric : string, optional
        Metric for distances in the kernel. Default is 'euclidean'. The callable should take two arrays as input and return one value indicating the distance between them.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.

    Examples
    --------
    # setup neighbor_params list with as many jobs as CPU cores and kd_tree neighbor search.
    >>> neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
    # initialize diffusion map object with the top two eigenvalues being computed, epsilon set to 0.1
    # and alpha set to 1.0.
    >>> mydmap = DiffusionMap(n_evecs = 2, epsilon = .1, alpha = 1.0, neighbor_params = neighbor_params)

    """

    def __init__(self, alpha=0.5, k=64, kernel_type='gaussian', epsilon='bgh', n_evecs=1, neighbor_params=None, metric='euclidean', metric_params=None):
        """
        Initializes Diffusion Map, sets parameters.
        """
        self.alpha = alpha
        self.k = k
        self.kernel_type = kernel_type
        self.epsilon = epsilon
        self.n_evecs = n_evecs
        self.neighbor_params = neighbor_params
        self.metric = metric
        self.metric_params = metric_params
        self.epsilon_fitted = None
        self.d = None

    def _compute_kernel(self, X):
        my_kernel = kernel.Kernel(kernel_type=self.kernel_type, k=self.k,
                                  epsilon=self.epsilon, neighbor_params=self.neighbor_params,
                                  metric=self.metric, metric_params=self.metric_params)
        my_kernel.fit(X)
        kernel_matrix = _symmetrize_matrix(my_kernel.compute(X))
        return kernel_matrix, my_kernel

    def _make_right_norm_vec(self, kernel_matrix, weights=None):
        # perform kde
        q = np.array(kernel_matrix.sum(axis=1)).ravel()
        # Apply right normalization
        right_norm_vec = np.power(q, -self.alpha)
        if weights is not None:
            right_norm_vec *= np.sqrt(weights)
        return q, right_norm_vec

    def _apply_normalizations(self, kernel_matrix, right_norm_vec):
        # Perform right normalization
        m = right_norm_vec.shape[0]
        Dalpha = sps.spdiags(right_norm_vec, 0, m, m)
        kernel_matrix = kernel_matrix * Dalpha

        # Perform  row (or left) normalization
        row_sum = kernel_matrix.sum(axis=1).transpose()
        n = row_sum.shape[1]
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
        P = Dalpha * kernel_matrix
        return P

    def _make_diffusion_coords(self, P):
        evals, evecs = spsl.eigs(P, k=(self.n_evecs+1), which='LM')
        ix = evals.argsort()[::-1][1:]
        evals = np.real(evals[ix])
        evecs = np.real(evecs[:, ix])
        dmap = np.dot(evecs, np.diag(evals))
        return dmap, evecs, evals

    def fit(self, X, weights=None):
        """
        Fits the data.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.
        weights : array-like, optional, shape(n_query)
            Values of a weight function for the data.  This effectively adds a drift term equivalent to the gradient of the log of weighting function to the final operator.

        Returns
        -------
        self : the object itself
        """
        kernel_matrix, my_kernel = self._compute_kernel(X)

        q, right_norm_vec = self._make_right_norm_vec(kernel_matrix, weights)
        P = self._apply_normalizations(kernel_matrix, right_norm_vec)
        dmap, evecs, evals = self._make_diffusion_coords(P)

        # Save constructed data.
        self.local_kernel = my_kernel
        self.epsilon_fitted = my_kernel.epsilon_fitted
        self.d = my_kernel.d
        self.data = X
        self.weights = weights
        self.kernel_matrix = kernel_matrix
        self.P = P
        self.q = q
        self.right_norm_vec = right_norm_vec
        self.evals = evals
        self.evecs = evecs
        self.dmap = dmap
        return self

    def transform(self, Y):
        """
        Performs Nystroem out-of-sample extension to calculate the values of the diffusion coordinates at each given point.

        Parameters
        ----------
        Y : array-like, shape (n_query, n_features)
            Data for which to perform the out-of-sample extension.

        Returns
        -------
        phi : numpy array, shape (n_query, n_eigenvectors)
            Transformed value of the given values.
        """
        # check if Y is equal to data. If yes, no computation needed.
        if np.array_equal(self.data, Y):
            return self.dmap
        else:
            # turn x into array if needed
            if (Y.ndim == 1):
                Y = Y[np.newaxis, :]
            # compute the values of the kernel matrix
            kernel_extended = self.local_kernel.compute(Y)
            P = self._apply_normalizations(kernel_extended, self.right_norm_vec)
            return P * self.evecs

    def fit_transform(self, X, weights=None):
        """
        Fits the data and returns diffusion coordinates.  equivalent to calling dmap.fit(X).transform(x).

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.
        weights : array-like, optional, shape (n_query)
            Values of a weight function for the data.  This effectively adds a drift term equivalent to the gradient of the log of weighting function to the final operator.

        Returns
        -------
        phi : numpy array, shape (n_query, n_eigenvectors)
            Transformed value of the given values.
        """
        self.fit(X, weights=weights)
        return self.dmap


def _symmetrize_matrix(K, mode='average'):
    """
    Symmetrizes a sparse kernel matrix.

    Parameters
    ----------
    K : scipy sparse matrix
        The sparse matrix to be symmetrized, with positive elements on the nearest neighbors.
    mode : string
        The method of symmetrization to be implemented.  Current options are 'average', 'and', and 'or'.

    Returns
    -------
    K_sym : scipy sparse matrix
        Symmetrized kernel matrix.
    """

    if mode == 'average':
        return 0.5*(K + K.transpose())
    elif mode == 'or':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K + dK
        return 0.5*K
    elif mode == 'and':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K - dK
        return 0.5*K
    else:
        raise ValueError('Did not understand symmetrization method')
