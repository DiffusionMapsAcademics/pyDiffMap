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
    epsilon : scalar, optional
        Length-scale parameter.
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    kernel_type : string, optional
        Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
    choose_eps : string, optional
        Method for choosing the epsilon.  Currently, the only option is 'fixed' (i.e. don't) or 'bgh' (Berry, Giannakis and Harlim)
    n_evecs : int, optional
        Number of diffusion map eigenvectors to return
    metric : string, optional
        Metric for distances in the kernel. Default is 'euclidean'. The callable should take two arrays as input and return one value indicating the distance between them.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.
    """

    def __init__(self, alpha=0.5, epsilon=1.0, k=64, kernel_type='gaussian', choose_eps='fixed', n_evecs=1, metric='euclidean', metric_params=None):
        """
        Initializes Diffusion Map, sets parameters
        """

        self.alpha = alpha
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.choose_eps = choose_eps
        self.k = k
        self.n_evecs = n_evecs
        self.metric = metric
        self.metric_params = metric_params

        return

    def fit(self, X):
        """
        Fits the data.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.

        Returns
        -------
        self : the object itself
        """
        self.data = X
        # compute kernel matrix
        my_kernel = kernel.Kernel(type=self.kernel_type, epsilon=self.epsilon,
                                  choose_eps=self.choose_eps, k=self.k, 
                                  metric=self.metric, metric_params=self.metric_params)
        self.local_kernel = my_kernel.fit(X)
        self.epsilon = my_kernel.epsilon
        kernel_matrix = _symmetrize_matrix(my_kernel.compute(X))

        # alpha normalization
        m = np.shape(X)[0]
        q = np.array(kernel_matrix.sum(axis=1)).ravel()
        Dalpha = sps.spdiags(np.power(q, -self.alpha), 0, m, m)
        kernel_matrix = Dalpha * kernel_matrix * Dalpha
        # save kernel density estimate for later
        self.q = q

        # row normalization
        row_sum = kernel_matrix.sum(axis=1).transpose()
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, m, m)
        P = Dalpha * kernel_matrix
        self.P = P

        # diagonalise and sort eigenvalues
        evals, evecs = spsl.eigs(P, k=(self.n_evecs+1), which='LM')
        ix = evals.argsort()[::-1]
        evals = evals[ix]
        evecs = evecs[:, ix]
        self.evals = np.real(evals[1:])
        self.evecs = np.real(evecs[:, 1:])
        self.dmap = np.dot(self.evecs, np.diag(self.evals))
        return self

#    def _get_bandwidth_fxn(self):
#        """
#
#        """
#        return

#    def _get_optimal_epsilon(self, scaled_distsq):
#        """
#
#        """
#        return

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
            # right normalization
            m = np.shape(self.data)[0]
            Dalpha = sps.spdiags(np.power(self.q, -self.alpha), 0, m, m)
            kernel_extended = kernel_extended * Dalpha
            # left normalization
            D = kernel_extended.sum(axis=1).transpose()
            Dalpha = sps.spdiags(np.power(D, -1), 0, np.shape(D)[1], np.shape(D)[1])
            P = Dalpha * kernel_extended
            return P * self.evecs

    def fit_transform(self, X):
        """
        Fits the data and returns diffusion coordinates.  equivalent to calling dmap.fit(X).transform(x).

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.

        Returns
        -------
        phi : numpy array, shape (n_query, n_eigenvectors)
            Transformed value of the given values.
        """
        self.fit(X)
        return self.dmap

#    def compute_dirichlet_basis(self):
#        """
#
#        """
#        return


#########################################

class TargetMeasureDiffusionMap(DiffusionMap):
    """
    Target Measure Diffusion Map object to be used in data analysis for fun and profit. Target Measure Diffusion Map
    creates (TMDmap) is algorithm which constructs a matrix on the data that approximates the differential operator
    .. math::  Lf = \Delta f + \nabla (\log \pi)\cdot\nabla f. The target density .. math:: \pi
    is evaluated on the data up to a normalization constant.


    """

    # def __init__(self, target_distribution, *args, **kwargs):
    #     self.target_distribution=target_distribution
    #     super().__init__(*args, **kwargs)
    #     return

    def fit_transform(self, X, target_distribution):
        self.fit(X, target_distribution)
        return self.dmap

    def fit(self, X, target_distribution):
        """
        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.
        Parameters
        ----------
        target distribution : array-like, shape n_query
            Target measure for TMDmap.
        Returns
        -------
        self : the object itself
        """
        self.data = X
        self.target_distribution = target_distribution
        # compute kernel matrix
        my_kernel = kernel.Kernel(type=self.kernel_type, epsilon=self.epsilon,
                                  choose_eps=self.choose_eps, k=self.k, 
                                  metric=self.metric, metric_params=self.metric_params)
        self.local_kernel = my_kernel.fit(X)
        self.local_kernel = my_kernel
        kernel_matrix = _symmetrize_matrix(my_kernel.compute(X))

        # alpha normalization: unbias point to obtain target measure
        m = np.shape(X)[0]
        # compute kernel density estimator: TODO use some other options (i.e. landmark KDE)
        q = np.array(kernel_matrix.sum(axis=1)).ravel()
        self.q = q
        # compute weights
        weights = np.zeros(m)
        for i in range(0, len(X)):
            weights[i] = np.sqrt(self.target_distribution[i]) / q[i]
        # save the weights
        self.weigths = weights

        D = sps.spdiags(weights, 0, m, m)
        Ktilde = kernel_matrix * D
        # row normalization
        Dalpha = sps.csr_matrix.sum(Ktilde, axis=1).transpose()
        Dtilde = sps.spdiags(np.power(Dalpha, -1), 0, m, m)

        P = Dtilde * Ktilde
        self.P = P

        # diagonalise and sort eigenvalues
        evals, evecs = spsl.eigs(P, k=(self.n_evecs+1), which='LM')
        ix = evals.argsort()[::-1]
        evals = evals[ix]
        evecs = evecs[:, ix]
        self.evals = np.real(evals[1:])
        self.evecs = np.real(evecs[:, 1:])
        self.dmap = np.dot(self.evecs, np.diag(self.evals))
        return self


TMDiffusionMap = TargetMeasureDiffusionMap


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
