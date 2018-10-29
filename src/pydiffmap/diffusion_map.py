# -*- coding: utf-8 -*-
"""
Routines and Class definitions for the diffusion maps algorithm.
"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import warnings
from . import kernel
from . import utils


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
    weight_fxn : callable or None, optional
        Callable function that take in two points (X_i and X_j), and outputs the value of the weight matrix at those points.
    density_fxn : callable or None, optional
        Callable function that take in X, and outputs the value of the density of X. Used instead of kernel density estimation in the normalisation.
    bandwidth_type: callable, number, string, or None, optional
        Type of bandwidth to use in the kernel.  If None (default), a fixed bandwidth kernel is used.  If a callable function, the data is passed to the function, and the bandwidth is output (note that the function must take in an entire dataset, not the points 1-by-1).  If a number, e.g. -.25, a kernel density estimate is performed, and the bandwidth is taken to be q**(input_number).  For a string input, the input is assumed to be an evaluatable expression in terms of the dimension d, e.g. "-1/(d+2)".  The dimension is then estimated, and the bandwidth is set to q**(evaluated input string).
    bandwidth_normalize: boolean, optional
        If true, normalize the final constructed transition matrix by the bandwidth as described in Berry and Harlim. [1]_
    oos : 'nystroem' or 'power', optional
        Method to use for out-of-sample extension.

    Examples
    --------
    # setup neighbor_params list with as many jobs as CPU cores and kd_tree neighbor search.
    >>> neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
    # initialize diffusion map object with the top two eigenvalues being computed, epsilon set to 0.1
    # and alpha set to 1.0.
    >>> mydmap = DiffusionMap(n_evecs = 2, epsilon = .1, alpha = 1.0, neighbor_params = neighbor_params)

    References
    ----------
    .. [1] T. Berry, and J. Harlim, Applied and Computational Harmonic Analysis 40, 68-96
       (2016).
    """

    def __init__(self, alpha=0.5, k=64, kernel_type='gaussian', epsilon='bgh', n_evecs=1, neighbor_params=None,
                 metric='euclidean', metric_params=None, weight_fxn=None, density_fxn=None, bandwidth_type=None,
                 bandwidth_normalize=False, oos='nystroem'):
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
        self.weight_fxn = weight_fxn
        self.bandwidth_normalize = bandwidth_normalize
        self.bandwidth_type = bandwidth_type
        if ((self.bandwidth_type is None) and (bandwidth_normalize is True)):
            warnings.warn('Bandwith normalization set to true, but no bandwidth function provided.  Setting to False.')
        self.oos = oos
        self.density_fxn = density_fxn

    def _build_kernel(self, X):
        my_kernel = kernel.Kernel(kernel_type=self.kernel_type, k=self.k,
                                  epsilon=self.epsilon, neighbor_params=self.neighbor_params,
                                  metric=self.metric, metric_params=self.metric_params,
                                  bandwidth_type=self.bandwidth_type)
        my_kernel.fit(X)
        kernel_matrix = utils._symmetrize_matrix(my_kernel.compute(X))
        return kernel_matrix, my_kernel

    def _compute_weights(self, X, kernel_matrix, Y):
        if self.weight_fxn is not None:
            return utils.sparse_from_fxn(X, kernel_matrix, self.weight_fxn, Y)
        else:
            return None

    def _make_right_norm_vec(self, kernel_matrix, q=None, bandwidths=None):
        if q is None:
            # perform kde
            q = np.array(kernel_matrix.sum(axis=1)).ravel()
            if bandwidths is not None:
                q /= bandwidths**2
        right_norm_vec = np.power(q, -self.alpha)
        return q, right_norm_vec

    def _right_normalize(self, kernel_matrix, right_norm_vec, weights):
        m = right_norm_vec.shape[0]
        Dalpha = sps.spdiags(right_norm_vec, 0, m, m)
        kernel_matrix = kernel_matrix * Dalpha
        if weights is not None:
            kernel_matrix = kernel_matrix.multiply(weights)
        return kernel_matrix

    def _left_normalize(self, kernel_matrix):
        row_sum = kernel_matrix.sum(axis=1).transpose()
        n = row_sum.shape[1]
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
        P = Dalpha * kernel_matrix
        return P

    def _build_generator(self, P, epsilon_fitted, bandwidths=None):
        print(epsilon_fitted, 'eps fitted')
        m, n = P.shape
        L = (P - sps.eye(m, n, k=(n - m))) / epsilon_fitted
        if bandwidths is not None:
            bw_diag = sps.spdiags(np.power(bandwidths, -2), 0, m, m)
            L = bw_diag * L
        return L

    def _make_diffusion_coords(self, L):
        evals, evecs = spsl.eigs(L, k=(self.n_evecs+1), which='LR')
        ix = evals.argsort()[::-1][1:]
        evals = np.real(evals[ix])
        evecs = np.real(evecs[:, ix])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1. / evals)))
        return dmap, evecs, evals

    def construct_Lmat(self, X):
        """
        Builds the transition matrix, but does NOT compute the eigenvectors.  This is useful for applications where the transition matrix itself is the object of interest.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to construct the diffusion map.

        Returns
        -------
        self : the object itself
        """
        kernel_matrix, my_kernel = self._build_kernel(X)
        weights = self._compute_weights(X, kernel_matrix, X)

        if self.density_fxn is not None:
            density = self.density_fxn(X)
        else:
            density = None
        q, right_norm_vec = self._make_right_norm_vec(kernel_matrix, q=density, bandwidths=my_kernel.bandwidths)
        P = self._right_normalize(kernel_matrix, right_norm_vec, weights)
        P = self._left_normalize(P)
        L = self._build_generator(P, my_kernel.epsilon_fitted, my_kernel.bandwidths)

        # Save data
        self.local_kernel = my_kernel
        self.epsilon_fitted = my_kernel.epsilon_fitted
        self.d = my_kernel.d
        self.data = X
        self.weights = weights
        self.kernel_matrix = kernel_matrix
        self.L = L
        self.q = q
        self.right_norm_vec = right_norm_vec
        return self

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
        self.construct_Lmat(X)
        dmap, evecs, evals = self._make_diffusion_coords(self.L)

        # Save constructed data.
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
        if np.array_equal(self.data, Y):
            return self.dmap
        else:
            # turn Y into 2D array if needed
            if (Y.ndim == 1):
                Y = Y[np.newaxis, :]

            if self.oos == "nystroem":
                return nystroem_oos(self, Y)
            elif self.oos == "power":
                return power_oos(self, Y)
            else:
                raise ValueError('Did not understand the OOS algorithm specified')

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


class TMDmap(DiffusionMap):
    """
    Implementation of the TargetMeasure diffusion map.  This provides a more convenient interface for some hyperparameter selection for the general diffusion object.  It takes the same parameters as the base Diffusion Map object.  However, rather than taking a weight function, it takes as input a change of measure function.

    Parameters
    ----------
    change_of_measure : callable, optional
        Function that takes in a point and evaluates the change-of-measure between the density otherwise stationary to the diffusion map and the desired density.
    """

    def __init__(self, alpha=0.5, k=64, kernel_type='gaussian', epsilon='bgh',
                 n_evecs=1, neighbor_params=None, metric='euclidean',
                 metric_params=None, change_of_measure=None, density_fxn=None,
                 bandwidth_type=None, bandwidth_normalize=False, oos='nystroem'):

        def weight_fxn(x_i, y_i):
            return np.sqrt(change_of_measure(y_i))

        super(TMDmap, self).__init__(alpha=alpha, k=k, kernel_type=kernel_type,
                                     epsilon=epsilon, n_evecs=n_evecs,
                                     neighbor_params=neighbor_params,
                                     metric=metric, metric_params=metric_params,
                                     density_fxn=density_fxn,
                                     bandwidth_type=bandwidth_type,
                                     bandwidth_normalize=bandwidth_normalize,
                                     weight_fxn=weight_fxn, oos=oos)


def nystroem_oos(dmap_object, Y):
    """
    Performs Nystroem out-of-sample extension to calculate the values of the diffusion coordinates at each given point.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    # check if Y is equal to data. If yes, no computation needed.
    # compute the values of the kernel matrix
    kernel_extended = dmap_object.local_kernel.compute(Y)
    weights = dmap_object._compute_weights(dmap_object.local_kernel.data, kernel_extended, Y)
    P = dmap_object._left_normalize(dmap_object._right_normalize(kernel_extended, dmap_object.right_norm_vec, weights))
    evals_p = dmap_object.local_kernel.epsilon_fitted * dmap_object.evals + 1.
    oos_evecs = P * dmap_object.dmap
    oos_dmap = np.dot(oos_evecs, np.diag(1. / evals_p))
    return oos_evecs


def power_oos(dmap_object, Y):
    """
    Performs out-of-sample extension to calculate the values of the diffusion coordinates at each given point using the power-like method.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    m = int(Y.shape[0])
    k_yx, y_bandwidths = dmap_object.local_kernel.compute(Y, return_bandwidths=True)  # Evaluate on ref points
    yy_right_norm_vec = dmap_object._make_right_norm_vec(k_yx, y_bandwidths)[1]

    k_yy_diag = dmap_object.local_kernel.kernel_fxn(0, dmap_object.epsilon_fitted)
    data_full = np.vstack([dmap_object.local_kernel.data, Y])
    k_full = sps.hstack([k_yx, sps.eye(m) * k_yy_diag])
    right_norm_full = np.hstack([dmap_object.right_norm_vec, yy_right_norm_vec])
    weights = dmap_object._compute_weights(data_full, k_full, Y)

    P = dmap_object._left_normalize(dmap_object._right_normalize(k_full, right_norm_full, weights))
    L = dmap_object._build_generator(P, dmap_object.epsilon_fitted,
                                             y_bandwidths)
    L_yx = L[:, :-m]
    L_yy = np.array(L[:, -m:].diagonal())
    adj_evals = dmap_object.evals - L_yy.reshape(-1, 1)
    dot_part = np.array(L_yx.dot(dmap_object.dmap))
    return (1. / adj_evals) * dot_part
