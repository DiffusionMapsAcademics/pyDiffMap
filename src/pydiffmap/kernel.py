"""
A class to implement diffusion kernels.
"""

import numbers
import numpy as np
import numexpr as ne
import scipy.sparse as sps
import warnings
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from six import string_types
from . import utils


class Kernel(object):
    """
    Class abstracting the evaluation of kernel functions on the dataset.

    Parameters
    ----------
    kernel_type : string or callable, optional
        Type of kernel to construct. Currently the only option is 'gaussian' (the default), but more will be implemented.
    epsilon : string, optional
        Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) 'bgh' (Berry, Giannakis and Harlim), and 'bgh_generous' ('bgh' method, with answer multiplied by 2.
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    neighbor_params : dict or None, optional
        Optional parameters for the nearest Neighbor search. See scikit-learn NearestNeighbors class for details.
    metric : string, optional
        Distance metric to use in constructing the kernel.  This can be selected from any of the scipy.spatial.distance metrics, or a callable function returning the distance.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.
    bandwidth_type: callable, number, string, or None, optional
        Type of bandwidth to use in the kernel.  If None (default), a fixed bandwidth kernel is used.  If a callable function, the data is passed to the function, and the bandwidth is output (note that the function must take in an entire dataset, not the points 1-by-1).  If a number, e.g. -.25, a kernel density estimate is performed, and the bandwidth is taken to be q**(input_number).  For a string input, the input is assumed to be an evaluatable expression in terms of the dimension d, e.g. "-1/(d+2)".  The dimension is then estimated, and the bandwidth is set to q**(evaluated input string).
    """

    def __init__(self, kernel_type='gaussian', epsilon='bgh', k=64, neighbor_params=None, metric='euclidean', metric_params=None, bandwidth_type=None):
        self.kernel_fxn = _parse_kernel_type(kernel_type)
        self.epsilon = epsilon
        self.k = k
        self.metric = metric
        self.metric_params = metric_params
        if neighbor_params is None:
            neighbor_params = {}
        self.neighbor_params = neighbor_params
        self.bandwidth_type = bandwidth_type
        self.d = None
        self.epsilon_fitted = None

    def build_bandwidth_fxn(self, bandwidth_type):
        """
        Parses an input string or function specifying the bandwidth.

        Parameters
        ----------
        bandwidth_fxn : string or number or callable
            Bandwidth to use.  If a number, taken to be the beta parameter in [1]_.
            If a string, taken to again be beta, but with an evaluatable
            expression as a function of the intrinsic dimension d, e.g. '1/(d+2)'.
            If a function, taken to be a function that outputs the bandwidth.

        References
        ----------
        .. [1] T. Berry, and J. Harlim, Applied and Computational Harmonic Analysis 40, 68-96
           (2016).
        """
        if self.bandwidth_type is None:
            return None
        elif callable(self.bandwidth_type):
            return self.bandwidth_type
        else:
            is_string = isinstance(self.bandwidth_type, string_types)
            is_number = isinstance(self.bandwidth_type, numbers.Number)
            if (is_string or is_number):
                kde_function, d = self._build_nn_kde()
                if is_string:
                    beta = ne.evaluate(self.bandwidth_type)
                elif is_number:
                    beta = self.bandwidth_type
                else:
                    raise Exception("Honestly, we shouldn't have gotten to this point in the code")
                bandwidth_fxn = lambda x: kde_function(x)**beta
                return bandwidth_fxn
            else:
                raise ValueError("Bandwidth Type was not a callable, string, or number.  Don't know what to make of it.")

    def _build_nn_kde(self, num_nearest_neighbors=8):
        my_nnkde = NNKDE(self.neigh, k=num_nearest_neighbors)
        my_nnkde.fit()
        bandwidth_fxn = lambda x: my_nnkde.compute(x)
        self.kde = my_nnkde
        return bandwidth_fxn, my_nnkde.d

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
        k0 = min(self.k, np.shape(X)[0])
        self.data = X
        # Construct Nearest Neighbor Tree
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.")
            self.neigh = NearestNeighbors(n_neighbors=k0,
                                          metric=self.metric,
                                          metric_params=self.metric_params,
                                          **self.neighbor_params)
        self.neigh.fit(X)
        self.bandwidth_fxn = self.build_bandwidth_fxn(self.bandwidth_type)
        self.bandwidths = self._compute_bandwidths(X)
        self.scaled_dists = self._get_scaled_distance_mat(self.data, self.bandwidths)
        self.choose_optimal_epsilon()
        return self

    def compute(self, Y=None, return_bandwidths=False):
        """
        Computes the sparse kernel matrix.

        Parameters
        ----------
        Y : array-like, shape (n_query, n_features), optional.
            Data against which to calculate the kernel values.  If not provided, calculates against the data provided in the fit.
        return_bandwidths : boolean, optional
            If True, also returns the computed bandwidth for each y point.

        Returns
        -------
        K : array-like, shape (n_query_X, n_query_Y)
            Values of the kernel matrix.
        y_bandwidths : array-like, shape (n_query_y)
            Bandwidth evaluated at each point Y.  Only returned if return_bandwidths is True.

        """
        if Y is None:
            Y = self.data
        if np.array_equal(Y, self.data):  # Avoid recomputing nearest neighbors unless needed.
            y_bandwidths = self.bandwidths
            K = self.scaled_dists
        else:
            # perform k nearest neighbour search on X and Y and construct sparse matrix
            # retrieve all nonzero elements and apply kernel function to it
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
        if y_bandwidths is not None:
            bw_x = np.power(self.bandwidths, 0.5)
            bw_y = np.power(y_bandwidths, 0.5)
            dists = _scale_by_bw(dists, bw_x, bw_y)
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
        elif ((epsilon == 'bgh') or (epsilon == 'bgh_generous')):  # Berry, Giannakis Harlim method.
            if (self.metric != 'euclidean'):  # TODO : replace with call to scipy metrics.
                warnings.warn('The BGH method for choosing epsilon assumes a euclidean metric.  However, the metric being used is %s.  Proceed at your own risk...' % self.metric)
            if self.scaled_dists is None:
                self.scaled_dists = self._get_scaled_distance_mat(self.data, self.bandwidths)
            self.epsilon_fitted, self.d = choose_optimal_epsilon_BGH(self.scaled_dists.data**2)
            if epsilon == 'bgh_generous':
                self.epsilon_fitted *= 2.
        else:
            raise ValueError("Method for automatically choosing epsilon was given as %s, but this was not recognized" % epsilon)
        return self


class NNKDE(object):
    """
    Class building a kernel density estimate with a variable bandwidth built from the k nearest neighbors.

    Parameters
    ----------
    neighbors : scikit-learn NearestNeighbors object
        NearestNeighbors object to use in constructing the KDE.
    k : int, optional
        Number of nearest neighbors to use in the construction of the bandwidth.  This must be less or equal to the number of nearest neighbors used by the nearest neighbor object.
    """

    def __init__(self, neighbors, k=8):
        self.neigh = neighbors
        self.kernel_fxn = _parse_kernel_type('gaussian')
        self.k = k

    def _reduce_nn(self, nn_graph, k):
        # gets the k nearest neighbors of an m nearest nearest graph,
        # where m >n
        sub_neighbors = []
        for row in nn_graph:
            dense_row = np.array(row[row.nonzero()]).ravel()
            sorted_ndxs = np.argpartition(dense_row, k-1)
            sorted_row = dense_row[sorted_ndxs[:k]]
            sub_neighbors.append(sorted_row)
        return np.array(sub_neighbors)

    def _build_bandwidth(self):
        dist_graph_vals = self._reduce_nn(self.dist_graph_sq, k=self.k-1)
        avg_sq_dist = np.array(dist_graph_vals.sum(axis=1)).ravel()
        self.bandwidths = np.sqrt(avg_sq_dist/(self.k-1)).ravel()

    def _choose_epsilon(self):
        # dist_graph_sq = self.neigh.kneighbors_graph(n_neighbors=self.neigh.n_neighbors-1, mode='distance')
        dist_graph_sq = self.dist_graph_sq.copy()
        n = dist_graph_sq.shape[0]
        dist_graph_sq = _scale_by_bw(dist_graph_sq, self.bandwidths, self.bandwidths)
        sq_dists = np.hstack([dist_graph_sq.data, np.zeros(n)])
        self.epsilon_fitted, self.d = choose_optimal_epsilon_BGH(sq_dists)

    def fit(self):
        """
        Fits the kde object to the data provided in the nearest neighbor object.
        """
        self.dist_graph_sq = self.neigh.kneighbors_graph(n_neighbors=self.neigh.n_neighbors-1,
                                                         mode='distance')
        self.dist_graph_sq.data = self.dist_graph_sq.data**2
        self._build_bandwidth()
        self._choose_epsilon()

    def compute(self, Y):
        """
        Computes the density at each query point in Y.

        Parameters
        ----------
        Y : array-like, shape (n_query, n_features)
            Data against which to calculate the kernel values.  If not provided, calculates against the data provided in the fit.


        Returns
        -------
        q : array-like, shape (n_query)
            Density evaluated at each point Y.
        """
        dist_bw = self.neigh.kneighbors_graph(Y, mode='distance', n_neighbors=self.k)
        dist_bw.data = dist_bw.data**2
        avg_sq_dist = np.array(dist_bw.sum(axis=1)).ravel()
        y_bandwidths = np.sqrt(avg_sq_dist/(self.k-1)).ravel()
        K = self.neigh.kneighbors_graph(Y, mode='distance')
        K.data = K.data**2
        K = _scale_by_bw(K, self.bandwidths, y_bandwidths)
        K.data /= 4. * self.epsilon_fitted
        K.data = np.exp(-K.data)
        density = np.array(K.mean(axis=1)).ravel()
        density /= y_bandwidths**self.d
        density /= (4 * np.pi * self.epsilon_fitted)**(self.d / 2.)
        return density


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
    # epsilon = np.max([np.exp(log_eps[max_loc]), np.exp(log_eps[max_loc+1])])
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
        def gaussian_kfxn(d, epsilon):
            return np.exp(-d**2 / (4. * epsilon))
        return gaussian_kfxn
    elif callable(kernel_type):
        return kernel_type
    else:
        raise("Error: Kernel type not understood.")


def _scale_by_bw(d_yx, bw_x, bw_y):
    """
    Scale a distance matrix with the bandwidth functions while retaining explicit zeros.
    Note that this reorders the indices in d_yx.

    Parameters
    ----------
    d_yx : scipy sparse matrix
        Sparse matrix whose i,j'th element corresponds to f(y_i, x_j)
    dw_x : numpy array
        Array of bandwidth values evaluated at each x_i
    dw_y : numpy array
        Array of bandwidth values evaluated at each y_i

    Returns
    ------
    scaled_d_yx : scipy sparse matrix
        Sparse matrix whose i,j'th element corresponds to f(y_i, x_j)/ bw[y_i] bw[x_j]
    """
    m, n = d_yx.shape
    x_bw_diag = sps.spdiags(np.power(bw_x, -1), 0, n, n)
    y_bw_diag = sps.spdiags(np.power(bw_y, -1), 0, m, m)
    row, col = utils._get_sparse_row_col(d_yx)
    inv_bw = sps.csr_matrix((np.ones(d_yx.data.shape), (row, col)), shape=d_yx.shape)
    inv_bw = y_bw_diag * inv_bw * x_bw_diag
    d_yx.sort_indices()
    inv_bw.sort_indices()
    d_yx.data = d_yx.data * inv_bw.data
    return d_yx
