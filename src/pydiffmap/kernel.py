"""
A class to implement diffusion kernels.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Kernel(object):
    """
    Class abstracting the evaluation of kernel functions on the dataset.

    Parameters
    ----------
    type : string, optional
        Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
    epsilon : scalar, optional
        Value of the length-scale parameter.
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    metric : string, optional
        Distance metric to use in constructing the kernel.  This can be selected from any of the scipy.spatial.distance metrics, or a callable function returning the distance.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.
    """

    def __init__(self, type='gaussian', epsilon=1.0, k=64, metric='euclidean', metric_params=None):
        self.type = type
        self.epsilon = epsilon
        self.metric = metric
        self.k = k

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
        self.neigh = NearestNeighbors(metric=self.metric).fit(X)
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
        K = self.neigh.kneighbors_graph(Y, n_neighbors=self.k0, mode='distance', metric = self.metric)
        # retrieve all nonzero elements and apply kernel function to it
        v = K.data
        if (self.type == 'gaussian'):
            K.data = np.exp(-v**2/self.epsilon )
        else:
            raise("Error: Kernel type not understood.")
        return K
