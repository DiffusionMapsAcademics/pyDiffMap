import numpy as np
from sklearn.neighbors import NearestNeighbors

class Kernel():

    def __init__(self, type='gaussian', distance = 'euclidean', epsilon = 1.0, k=64):


        self.type = type
        self.epsilon = epsilon
        self.distance = distance
        self.k = k

    def compute(self, X, Y=None):
        """
        compute sparse kernel matrix
        input:  X = (n,d) numpy array of X data points,
                Y = (m,d) numpy array of Y data points,
                rows correspond to different observations,
                columns to different variables
        output: sparse n x m kernel matrix k(X,Y).
        """
        compute_self = False
        if Y is None:
            Y = X
            compute_self = True
        # perform k nearest neighbour search on X and Y and construct sparse matrix
        neigh = NearestNeighbors(metric=self.distance)
        k0 = min(self.k, np.shape(Y)[0])
        A = neigh.fit(Y).kneighbors_graph(X,n_neighbors=k0, mode='distance')
        # retrieve all nonzero elements and apply kernel function to it
        v = A.data
        if (self.type=='gaussian'):
            A.data = np.exp(-v**2/self.epsilon)
        else:
            raise("Error: Kernel type not understood.")
        # symmetrize
        if (compute_self == True):
            A = 0.5*(A+A.transpose())

        return A
