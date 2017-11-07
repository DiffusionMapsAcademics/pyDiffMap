# -*- coding: utf-8 -*-
"""Routines and Class definitions for the
diffusion maps algorithm.

@authors: Erik, Zofia, Ralf, Lorenzo

"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import kernel as kern


class DiffusionMap(object):
    """

    """

    def __init__(self, alpha = 0.5, epsilon = 1.0, k=64, kernel_type = 'gaussian', choose_eps = 'auto', n_evecs = 1):
        """
        Initializes Diffusion Map Params.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.choose_eps = choose_eps
        self.k = k
        self.n_evecs = n_evecs
        return

    def fit(self,X):
        """
        Fits the data.
        input: X = (n,d) numpy array. Rows correspond to different observations,
        columns to different variables.
        creates the following new attributes:
        self.data   =   a copy of the data X
        self.q      =   the KDE of sampling density q as a row vector
        self.evals  =   eigenvalues 1 to n_evecs
        self.evecs  =   eigenvectors 1 to n_evecs
        self.dmap   =   diffusion map coordinates (eigenvectors scaled by eigenvalues)
                        1 to n_evecs
        """
        # save the locations of data points into the class.
        # Not pretty, but needed for the nystroem extension
        self.data = X
        #ToDo: compute epsilon automatically
        #if (choose_eps=='auto'):
            #self.epsilon = choose_epsilon(X)
        #compute kernel matrix
        kernel = kern.Kernel(type=self.kernel_type, epsilon = self.epsilon, k=self.k).compute(X)
        #alpha normalization
        m = np.shape(X)[0];
        q = sps.csr_matrix.sum(kernel, axis=1).transpose();
        Dalpha = sps.spdiags(np.power(q,-self.alpha), 0, m, m)
        kernel = Dalpha * kernel * Dalpha;
        #save kernel density estimate for later
        self.q = q
        #row normalization
        D = sps.csr_matrix.sum(kernel, axis=1).transpose();
        Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
        P = Dalpha * kernel;
        #diagonalise and sort eigenvalues
        evals, evecs = spsl.eigs(P, k=(self.n_evecs+1), which='LM')
        ix = evals.argsort()[::-1]
        evals = evals[ix]
        evecs = evecs[:,ix]
        self.evals = np.real(evals[1:])
        self.evecs = np.real(evecs[:,1:])
        self.dmap = np.dot(self.evecs,np.diag(self.evals))
        return self

    def _get_bandwidth_fxn(self):
        """

        """
        return

    def _get_optimal_epsilon(self, scaled_distsq):
        """

        """
        return

    def transform(self, x):
        """
        computes diffusion map at location x.
        input: x = d-dim vector or (n,d) numpy array. rows correspond to different
        observations, columns to different variables.
        returns the diffusion map self.dmap evaluated at the point(s) x by
        the nystroem extension method.
        output: diffusion map evecs scaled by evals evaluated at x.
        """
        # turn x into array if needed
        if (x.ndim==1):
            x = x[np.newaxis,:]
        #compute the kernel k(x,X). x is the query point, X the data points.
        kernel_extended = kern.Kernel(type=self.kernel_type, epsilon = self.epsilon, k=self.k).compute(x, self.data)
        #right normalization
        m = np.shape(X)[0];
        Dalpha = sps.spdiags(np.power(self.q,-self.alpha), 0, m, m)
        kernel_extended = kernel_extended * Dalpha;
        #left normalization
        D = sps.csr_matrix.sum(kernel_extended, axis=1).transpose();
        Dalpha = sps.spdiags(np.power(D,-1), 0, np.shape(D)[1], np.shape(D)[1])
        P = Dalpha * kernel_extended;
        return P * self.evecs

    def fit_transform(self, X):
        """
        fits the data and returns diffusion coordinates.
        """
        self.fit(X);
        return self.dmap

    def compute_dirichlet_basis(self):
        """

        """
        return



def get_optimal_epsilon_BH(scaled_distsq, epses=2.**np.arange(-40, 41)):
    """
    Calculates the optimal bandwidth for kernel density estimation, according to the algorithm of Berry and Harlim.
    """
    return
