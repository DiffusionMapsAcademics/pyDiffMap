"""
A class to implement diffusion kernels.
"""

import numbers
import numpy as np
import scipy.sparse as sps
import warnings
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from . import utils


class Kernel(object):
    """
    Class abstracting the evaluation of kernel functions on the dataset.

    Parameters
    ----------
    kernel_type : string or callable, optional
        Type of kernel to construct. Currently the only option is 'gaussian' (the default), but more will be implemented.
    epsilon : string, optional
        Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) or 'bgh' (Berry, Giannakis and Harlim).
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    neighbor_params : dict or None, optional
        Optional parameters for the nearest Neighbor search. See scikit-learn NearestNeighbors class for details.
    metric : string, optional
        Distance metric to use in constructing the kernel.  This can be selected from any of the scipy.spatial.distance metrics, or a callable function returning the distance.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.
    """

    def __init__(self, kernel_type='gaussian', epsilon='bgh', k=64, neighbor_params=None, metric='euclidean', metric_params=None, bandwidth_fxn=None):
        self.kernel_fxn = _parse_kernel_type(kernel_type)
        self.epsilon = epsilon
        self.k = k
        self.metric = metric
        self.metric_params = metric_params
        if neighbor_params is None:
            neighbor_params = {}
        self.neighbor_params = neighbor_params
        self.bandwidth_fxn = bandwidth_fxn
        self.d = None
        self.epsilon_fitted = None

    def _compute_bandwidths(self, X):
        if self.bandwidth_fxn is not None:
            return self.bandwidth_fxn(X)
        else:
            return None

    def fit(self, X):
        """
        Fits the kernel to the data X, constructing the nearest neighbor tree.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Data upon which to fit the nearest neighbor tree.

        Returns
        -------
        self : the object itself
        """
        self.k0 = min(self.k, np.shape(X)[0])
        self.data = X
        # Construct Nearest Neighbor Tree
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.")
            self.neigh = NearestNeighbors(n_neighbors=self.k,
                                          metric=self.metric,
                                          metric_params=self.metric_params,
                                          **self.neighbor_params)
        self.neigh.fit(X)
        self.bandwidths = self._compute_bandwidths(X)
        self.choose_optimal_epsilon()
        return self

    def compute(self, Y=None, return_bandwidths=False):
        """
        Computes the sparse kernel matrix.

        Parameters
        ----------
        Y : array-like, shape (n_query, n_features), optional.
            Data against which to calculate the kernel values.  If not provided, calculates against the data provided in the fit.

        Returns
        -------
        K : array-like, shape (n_query_X, n_query_Y)
            Values of the kernel matrix.

        """
        if Y is None:
            Y = self.data
        # # perform k nearest neighbour search on X and Y and construct sparse matrix
        # # retrieve all nonzero elements and apply kernel function to it
        y_bandwidths = self._compute_bandwidths(Y)
        K = self._get_scaled_distance_mat(Y, y_bandwidths=y_bandwidths)
        K.data = self.kernel_fxn(K.data, self.epsilon_fitted)
        if return_bandwidths:
            return K, y_bandwidths
        else:
            return K

    def _get_scaled_distance_mat(self, Y, y_bandwidths=None):
        # Scales distance matrix by (rho(x) rho(y))^1/2, where rho is the
        # bandwidth.
        dists = self.neigh.kneighbors_graph(Y, mode='distance')
        m, n = dists.shape
        if y_bandwidths is not None:
            x_bw_diag = sps.spdiags(np.power(self.bandwidths, -0.5), 0, n, n).tocsr()
            y_bw_diag = sps.spdiags(np.power(y_bandwidths, -0.5), 0, m, m).tocsr()

            # Scale distances by bandwidth. This complement procedure is needed
            # to ensure that explicit zeros are preserved, which doesn't happen
            # with regular sparse matrix multiplication.
            row, col = utils._get_sparse_row_col(dists)
            inv_bw = sps.csr_matrix((np.ones(dists.data.shape), (row, col)), shape=dists.shape)
            inv_bw = y_bw_diag * inv_bw * x_bw_diag
            dists.sort_indices()
            inv_bw.sort_indices()
            dists.data = dists.data * inv_bw.data
        return dists

    def choose_optimal_epsilon(self, epsilon=None):
        """
        Chooses the optimal value of epsilon and automatically detects the
        dimensionality of the data.

        Parameters
        ----------
        epsilon : string or scalar, optional
            Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) or 'bgh' (Berry, Giannakis and Harlim).

        Returns
        -------
        self : the object itself
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Choose Epsilon according to method provided.
        if isinstance(epsilon, numbers.Number):  # if user provided.
            self.epsilon_fitted = epsilon
            return self
        elif epsilon == 'bgh':  # Berry, Giannakis Harlim method.
            scaled_dists = self._get_scaled_distance_mat(self.data, self.bandwidths)
            if (self.metric != 'euclidean'):  # TODO : replace with call to scipy metrics.
                warnings.warn('The BGH method for choosing epsilon assumes a euclidean metric.  However, the metric being used is %s.  Proceed at your own risk...' % self.metric)
            self.epsilon_fitted, self.d = choose_optimal_epsilon_BGH(scaled_dists.data**2)
        else:
            raise ValueError("Method for automatically choosing epsilon was given as %s, but this was not recognized" % epsilon)
        return self


def choose_optimal_epsilon_BGH(scaled_distsq, epsilons=None):
    """
    Calculates the optimal epsilon for kernel density estimation according to
    the criteria in Berry, Giannakis, and Harlim.

    Parameters
    ----------
    scaled_distsq : numpy array
        Values for scaled distance squared values, in no particular order or shape. (This is the exponent in the Gaussian Kernel, aka the thing that gets divided by epsilon).
    epsilons : array-like, optional
        Values of epsilon from which to choose the optimum.  If not provided, uses all powers of 2. from 2^-40 to 2^40

    Returns
    -------
    epsilon : float
        Estimated value of the optimal length-scale parameter.
    d : int
        Estimated dimensionality of the system.

    Notes
    -----
    This code explicitly assumes the kernel is gaussian, for now.

    References
    ----------
    The algorithm given is based on [1]_.  If you use this code, please cite them.

    .. [1] T. Berry, D. Giannakis, and J. Harlim, Physical Review E 91, 032915
       (2015).
    """
    if epsilons is None:
        epsilons = 2**np.arange(-40., 41., 1.)

    epsilons = np.sort(epsilons).astype('float')
    log_T = [logsumexp(-scaled_distsq/(4. * eps)) for eps in epsilons]
    log_eps = np.log(epsilons)
    log_deriv = np.diff(log_T)/np.diff(log_eps)
    max_loc = np.argmax(log_deriv)
    epsilon = np.exp(log_eps[max_loc])
    d = np.round(2.*log_deriv[max_loc])
    return epsilon, d


def _parse_kernel_type(kernel_type):
    """
    Parses an input string or function specifying the kernel.

    Parameters
    ----------
    kernel_type : string or callable
        Type of kernel to construct. Currently the only option is 'gaussian' or
        a user provided function.  If set to a user defined function, it should
        take in two arguments: in order, a vector of distances between two
        samples, and a length-scale parameter epsilon.  The units on epsilon
        should be distance squared.

    Returns
    -------
    kernel_fxn : callable
        Function that takes in the distance and length-scale parameter, and outputs the value of the kernel.
    """
    if kernel_type.lower() == 'gaussian':
        return lambda d, epsilon: np.exp(-d**2 / (4. * epsilon))
    elif callable(kernel_type):
        return kernel_type
    else:
        raise("Error: Kernel type not understood.")
