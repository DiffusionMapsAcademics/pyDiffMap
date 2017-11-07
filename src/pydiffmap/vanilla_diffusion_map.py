# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
vanilla diffusion maps algorithm (Coifman & Lafon 2006).

@author: Zofia

"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from . import kernel
from . diffusion_map import  DiffusionMap

class VanillaDiffusionMap(DiffusionMap):
    """Compute vanilla diffusion maps


    """

    def __init__(self):
        """
        Initializes Diffusion Map Params.
        """

        self.kernel=kernel.Kernel()

        return

    def fit(self,X):
        """
        Fits the data.

        """

        self.graphLaplacian = self._compute_normalized_graph_laplacian( X )


        return self

    def _compute_normalized_graph_laplacian(self, X):
        """
        Compute normalized graph Laplacian for vanilla diffusion map (Coifman & Lafon 2006)
        input: X = (n,d) numpy array of X data points
        output: L = (n,n) numpy array of normalized graph Laplacian
        """
        self.kernelMatrix = self.kernel.compute(X)
        kernel = self.kernelMatrix

        alpha = 0.5;
        m = np.shape(X)[0];
        D = sps.csr_matrix.sum(kernel, axis=0);
        Dalpha = sps.spdiags(np.power(D,-alpha), 0, m, m)
        kernel = Dalpha * kernel * Dalpha;
        D = sps.csr_matrix.sum(kernel, axis=0);
        Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
        L = Dalpha * kernel;

        return L

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

        """



        return

    def compute_dirichlet_basis(self):
        """

        """
        return



def get_optimal_epsilon_BH(scaled_distsq, epses=2.**np.arange(-40, 41)):
    """
    Calculates the optimal bandwidth for kernel density estimation, according to the algorithm of Berry and Harlim.
    """
    return
