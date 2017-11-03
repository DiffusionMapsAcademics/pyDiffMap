# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
diffusion maps algorithm.

@author: Erik

"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


class DiffusionMap(object):
    """
        
    """

    def __init__(self):
        """
        Initializes Diffusion Map Params.
        """
        return

    def fit(self,X):
        """
        Fits the data.

        """
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

