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
    Diffusion Map object for data analysis

    Parameters
    ----------
    kernel_object : Kernel object.
        Kernel object that outputs the values of the kernel.  Must have the method .fit(X) and .compute() methods.
        Any epsilon desired for normalization should be stored at kernel_object.epsilon_fitted and any bandwidths
        should be located at kernel_object.bandwidths.
    alpha : scalar, optional
        Exponent to be used for the left normalization in constructing the diffusion map.
    n_evecs : int, optional
        Number of diffusion map eigenvectors to return
    weight_fxn : callable or None, optional
        Callable function that take in a point, and outputs the value of the weight matrix at those points.
    density_fxn : callable or None, optional
        Callable function that take in X, and outputs the value of the density of X. Used instead of kernel density estimation in the normalisation.
    bandwidth_normalize: boolean, optional
        If true, normalize the final constructed transition matrix by the bandwidth as described in Berry and Harlim. [1]_
    oos : 'nystroem' or 'power', optional
        Method to use for out-of-sample extension.

    References
    ----------
    .. [1] T. Berry, and J. Harlim, Applied and Computational Harmonic Analysis 40, 68-96
       (2016).
    """

    def __init__(self, kernel_object, alpha=0.5, n_evecs=1,
                 weight_fxn=None, density_fxn=None,
                 bandwidth_normalize=False, oos='nystroem'):
        """
        Initializes Diffusion Map, sets parameters.
        """
        self.alpha = alpha
        self.n_evecs = n_evecs
        self.epsilon_fitted = None
        self.weight_fxn = weight_fxn
        self.bandwidth_normalize = bandwidth_normalize
        self.oos = oos
        self.density_fxn = density_fxn
        self.local_kernel = kernel_object

    @classmethod
    def from_sklearn(cls, alpha=0.5, k=64, kernel_type='gaussian', epsilon='bgh', n_evecs=1, neighbor_params=None,
                     metric='euclidean', metric_params=None, weight_fxn=None, density_fxn=None, bandwidth_type=None,
                     bandwidth_normalize=False, oos='nystroem'):
        """
        Builds the diffusion map using a kernel constructed using the Scikit-learn nearest neighbor object.
        Parameters are largely the same as the constructor, but in place of the kernel object it take
        the following parameters.

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors over which to construct the kernel.
        kernel_type : string, optional
            Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
        epsilon: string or scalar, optional
            Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) 'bgh' (Berry, Giannakis and Harlim), and 'bgh_generous' ('bgh' method, with answer multiplied by 2.
        neighbor_params : dict or None, optional
            Optional parameters for the nearest Neighbor search. See scikit-learn NearestNeighbors class for details.
        metric : string, optional
            Metric for distances in the kernel. Default is 'euclidean'. The callable should take two arrays as input and return one value indicating the distance between them.
        metric_params : dict or None, optional
            Optional parameters required for the metric given.
        bandwidth_type: callable, number, string, or None, optional
            Type of bandwidth to use in the kernel.  If None (default), a fixed bandwidth kernel is used.  If a callable function, the data is passed to the function, and the bandwidth is output (note that the function must take in an entire dataset, not the points 1-by-1).  If a number, e.g. -.25, a kernel density estimate is performed, and the bandwidth is taken to be q**(input_number).  For a string input, the input is assumed to be an evaluatable expression in terms of the dimension d, e.g. "-1/(d+2)".  The dimension is then estimated, and the bandwidth is set to q**(evaluated input string).

        Examples
        --------
        # setup neighbor_params list with as many jobs as CPU cores and kd_tree neighbor search.
        >>> neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
        # initialize diffusion map object with the top two eigenvalues being computed, epsilon set to 0.1
        # and alpha set to 1.0.
        >>> mydmap = DiffusionMap.from_sklearn(n_evecs = 2, epsilon = .1, alpha = 1.0, neighbor_params = neighbor_params)

        References
        ----------
        .. [1] T. Berry, and J. Harlim, Applied and Computational Harmonic Analysis 40, 68-96
           (2016).
        """

        buendia = kernel.Kernel(kernel_type=kernel_type, k=k, epsilon=epsilon, neighbor_params=neighbor_params, metric=metric, metric_params=metric_params, bandwidth_type=bandwidth_type)
        dmap = cls(buendia, alpha=alpha, n_evecs=n_evecs, weight_fxn=weight_fxn, density_fxn=density_fxn, bandwidth_normalize=bandwidth_normalize, oos=oos)
        # if ((bandwidth_type is None) and (bandwidth_normalize is True)):
        #     warnings.warn('Bandwith normalization set to true, but no bandwidth function provided.  Setting to False.')
        return dmap

    def _build_kernel(self, X, my_kernel):
        my_kernel.fit(X)
        kernel_matrix = utils._symmetrize_matrix(my_kernel.compute())
        return kernel_matrix, my_kernel

    def _compute_weights(self, X):
        if self.weight_fxn is not None:
            N = np.shape(X)[0]
            return np.array([self.weight_fxn(Xi) for Xi in X]).reshape(N)
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
            weight_mat = sps.spdiags(weights, 0, m, m)
            kernel_matrix = kernel_matrix * weight_mat
        return kernel_matrix

    def _left_normalize(self, kernel_matrix):
        row_sum = kernel_matrix.sum(axis=1).transpose()
        n = row_sum.shape[1]
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
        P = Dalpha * kernel_matrix
        return P

    def _build_generator(self, P, epsilon_fitted, bandwidths=None, bandwidth_normalize=False):
        m, n = P.shape
        L = (P - sps.eye(m, n, k=(n - m))) / epsilon_fitted
        if bandwidth_normalize:
            if bandwidths is not None:
                bw_diag = sps.spdiags(np.power(bandwidths, -2), 0, m, m)
                L = bw_diag * L
            else:
                warnings.warn('Bandwith normalization set to true, but no bandwidth function was found in normalization.  Not performing normalization')

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
        kernel_matrix, my_kernel = self._build_kernel(X, self.local_kernel)
        weights = self._compute_weights(X)

        if self.density_fxn is not None:
            density = self.density_fxn(X)
        else:
            density = None
        try:
            bandwidths = my_kernel.bandwidths
        except AttributeError:
            bandwidths = None

        q, right_norm_vec = self._make_right_norm_vec(kernel_matrix, q=density, bandwidths=bandwidths)
        P = self._right_normalize(kernel_matrix, right_norm_vec, weights)
        P = self._left_normalize(P)
        L = self._build_generator(P, my_kernel.epsilon_fitted, bandwidths, bandwidth_normalize=self.bandwidth_normalize)

        # Save data
        self.local_kernel = my_kernel
        self.epsilon_fitted = my_kernel.epsilon_fitted
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

        def weight_fxn(y_i):
            return np.sqrt(change_of_measure(y_i))

        buendia = kernel.Kernel(kernel_type=kernel_type, k=k, epsilon=epsilon, neighbor_params=neighbor_params, metric=metric, metric_params=metric_params, bandwidth_type=bandwidth_type)

        super(TMDmap, self).__init__(buendia, alpha=alpha, n_evecs=n_evecs, weight_fxn=weight_fxn, density_fxn=density_fxn, bandwidth_normalize=bandwidth_normalize, oos=oos)


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
    weights = dmap_object._compute_weights(dmap_object.local_kernel.data)
    P = dmap_object._left_normalize(dmap_object._right_normalize(kernel_extended, dmap_object.right_norm_vec, weights))
    oos_evecs = P * dmap_object.dmap
    # evals_p = dmap_object.local_kernel.epsilon_fitted * dmap_object.evals + 1.
    # oos_dmap = np.dot(oos_evecs, np.diag(1. / evals_p))
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
    weights = dmap_object._compute_weights(data_full)

    P = dmap_object._left_normalize(dmap_object._right_normalize(k_full, right_norm_full, weights))
    L = dmap_object._build_generator(P, dmap_object.epsilon_fitted, y_bandwidths)
    L_yx = L[:, :-m]
    L_yy = np.array(L[:, -m:].diagonal())
    adj_evals = dmap_object.evals - L_yy.reshape(-1, 1)
    dot_part = np.array(L_yx.dot(dmap_object.dmap))
    return (1. / adj_evals) * dot_part
