"""
A class to implement diffusion kernels.
"""

import numbers
import numpy as np
import warnings
from scipy.misc import logsumexp
from sklearn.neighbors import NearestNeighbors


class Kernel(object):
    """
    Class abstracting the evaluation of kernel functions on the dataset.

    Parameters
    ----------
    type : string, optional
        Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
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

    def __init__(self, kernel_type='gaussian', epsilon='bgh', k=64, neighbor_params=None, metric='euclidean', metric_params=None):
        self.type = kernel_type
        self.epsilon = epsilon
        self.k = k
        self.metric = metric
        self.metric_params = metric_params
        if neighbor_params is None:
            neighbor_params = {}
        self.neighbor_params = neighbor_params
        self.d = None
        self.epsilon_fitted = None

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
        self.choose_optimal_epsilon()
        return self

    def compute(self, Y=None):
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
        # perform k nearest neighbour search on X and Y and construct sparse matrix
        K = self.neigh.kneighbors_graph(Y, mode='distance')
        # retrieve all nonzero elements and apply kernel function to it
        v = K.data
        if (self.type == 'gaussian'):
            K.data = np.exp(-v**2/self.epsilon_fitted)
        else:
            raise("Error: Kernel type not understood.")
        return K

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
            dists = self.neigh.kneighbors_graph(self.data, mode='distance').data
            sq_distances = dists**2
            if (self.metric != 'euclidean'):  # TODO : replace with call to scipy metrics.
                warnings.warn('The BGH method for choosing epsilon assumes a euclidean metric.  However, the metric being used is %s.  Proceed at your own risk...' % self.metric)
            self.epsilon_fitted, self.d = choose_optimal_epsilon_BGH(sq_distances)
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
    Erik sez : I have a suspicion that the derivation here explicitly assumes that
    the kernel is Gaussian.  However, I'm not sure.  Also, we should perhaps replace
    this with some more intelligent optimization routine.  Here, I'm just
    picking from several values and choosin the best.

    References
    ----------
    The algorithm given is based on [1]_.  If you use this code, please cite them.

    .. [1] T. Berry, D. Giannakis, and J. Harlim, Physical Review E 91, 032915
       (2015).
    """
    if epsilons is None:
        epsilons = 2**np.arange(-40., 41., 1.)

    epsilons = np.sort(epsilons).astype('float')
    log_T = [logsumexp(-scaled_distsq/eps) for eps in epsilons]
    log_eps = np.log(epsilons)
    log_deriv = np.diff(log_T)/np.diff(log_eps)
    max_loc = np.argmax(log_deriv)
    epsilon = np.exp(log_eps[max_loc])
    d = np.round(2.*log_deriv[max_loc])
    return epsilon, d
