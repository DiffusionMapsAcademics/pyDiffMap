"""A class to implement diffusion kernels.

@authors: Erik, Zofia, Ralf, Lorenzo

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

class Kernel():

    def __init__(self, type='gaussian', distance = 'euclidean', epsilon = 1.0, k=64):


        self.type = type
        self.epsilon = epsilon
        self.distance = distance
        self.k = k

    def fit(self, X):
        self.k0 = min(self.k, np.shape(X)[0])
        self.data = X
        self.neigh = NearestNeighbors(metric=self.distance).fit(X)
        return self

    def compute(self, Y=None):
        """
        compute sparse kernel matrix
        input:  X = (n,d) numpy array of X data points,
                Y = (m,d) numpy array of Y data points,
                rows correspond to different observations,
                columns to different variables
        output: sparse n x m kernel matrix k(X,Y).
        """
        # perform k nearest neighbour search on X and Y and construct sparse matrix
        A = self.neigh.kneighbors_graph(Y,n_neighbors=self.k0, mode='distance')
        # retrieve all nonzero elements and apply kernel function to it
        v = A.data
        if (self.type=='gaussian'):
            A.data = np.exp(-v**2/self.epsilon)
        else:
            raise("Error: Kernel type not understood.")
        return A
