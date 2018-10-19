# -*- coding: utf-8 -*-
"""
Utilities for constructing diffusion maps.
"""
import numpy as np
import scipy.sparse as sps


def lookup_fxn(x, vals):
    """
    Builds a simple function that acts as a lookup table.  Useful for
    constructing bandwidth and weigth functions from existing values.

    Parameters
    ----------
    x : iterable
        values to input for the function
    vals : iterable
        Output values for the function.  Must be of the same length as x.

    Returns
    -------
    lf : function
        A function that, when input a value in x, outputs the corresponding
        value in vals.
    """
    # Build dictionary
    lookup = {}
    for i in range(len(x)):
        lookup[str(x[i])] = vals[i]

    # Define and return lookup function
    def lf(xi):
        return lookup[str(xi)]

    return lf


def sparse_from_fxn(X, K, function, Y=None):
    """
    For a function f, constructs a sparse matrix where each element is
    f(Y_i, X_j) with the same sparsity structure as the matrix K.

    Parameters
    ----------
    neighbors : scikit-learn NearestNeighbors object
        Data structure containing the nearest neighbor information.
        X values are drawn from the data in this object.
    function : function
        Function to apply to the pair Y_i, X_j.  Must take only two arguments
        and return a number.
    Y : iterable or None
        Values corresponding to each column of the matrix.  If None, defaults
        to the data in the neighbors object.

    Returns
    -------
    M : scipy sparse csr matrix
        Matrix with elements f(Y_i, X_j) for nearest neighbors, and zero
        otherwise.  Here Y_i is the i'th datapoint in Y, and X_j is the
        j'th datapoint in the NearestNeighbors object.
    """
    if Y is None:
        Y = X
    row, col = _get_sparse_row_col(K)

    fxn_vals = []
    for i, j in zip(row, col):
        fxn_vals.append(function(Y[i], X[j]))
    fxn_vals = np.array(fxn_vals)
    return sps.csr_matrix((fxn_vals, (row, col)), shape=K.shape)


def _get_sparse_row_col(sparse_mat):
    sparse_mat = sparse_mat.tocoo()
    return sparse_mat.row, sparse_mat.col


def _symmetrize_matrix(K, mode='or'):
    """
    Symmetrizes a sparse kernel matrix.

    Parameters
    ----------
    K : scipy sparse matrix
        The sparse matrix to be symmetrized, with positive elements on the nearest neighbors.
    mode : string
        The method of symmetrization to be implemented.  Current options are 'average', 'and', and 'or'.

    Returns
    -------
    K_sym : scipy sparse matrix
        Symmetrized kernel matrix.
    """

    if mode == 'average':
        return 0.5*(K + K.transpose())
    elif mode == 'or':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K + dK
        return 0.5*K
    elif mode == 'and':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K - dK
        return 0.5*K
    else:
        raise ValueError('Did not understand symmetrization method')
