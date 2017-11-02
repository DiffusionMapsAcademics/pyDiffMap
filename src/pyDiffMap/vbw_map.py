# -*- coding: utf-8 -*-
"""Routines and Class definitions for constructing basis sets using the
diffusion maps algorithm.

@author: Erik

"""
from __future__ import absolute_import, division, print_function

import numbers
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from ._NumericStringParser import _NumericStringParser


class DiffusionAtlas(object):
    """The diffusion atlas is a factory object for constructing diffusion map
    bases with various boundary conditions."""

    def __init__(self, nneighbors=600, d=None,
                 alpha='0', beta='-1/d', epses=2.**np.arange(-40, 41),
                 rho_norm=False, metric='euclidean', metric_params=None, verbosity=0):
        """Constructs the factory object.  The factory object can then be
        called to make diffusion map bases of various boundary conditions.

        Parameters
        ----------
        nneighbors : int or None, optional
            Number of neighbors to include in constructing the diffusion map.  If None, all neighbors are used.  Default is 600 neighbors
        d : int or None, optional
            Dimension of the system. If None and alpha or beta settings require the dimensionality,
            the dimension is estimated using the kernel density estimate,
            if a kernel density estimate is performed.
        alpha : float or string, optional
            Parameter for left normalization of the Diffusion map.
            Either a float, or a string that can be interpreted as a mathematical expression.
            The variable "d" stands for dimension, so "1/d" sets the alpha to one over the system dimension.
            Default is 0
        beta : float or string, optional
            Parameter for constructing the bandwidth function for the Diffusion map.  If rho is None, the bandwidth function will be set to q_\epsilon^beta, where q_\epsilon is an estimate of the density constructed using a kernel density estimate.  If rho is provided, this parameter is unused.  As with alpha, this will interpret strings that are evaluatable expressions.  Default is -1/(d+2)
        epses: float or 1d array, optional
            Bandwidth constant to use.  If float, uses that value for the bandwidth.  If array, performs automatic bandwidth detection according to the algorithm given by Berry and Giannakis and Harlim.  Default is all powers of 2 from 2^-40 to 2^40.
        rho_norm : bool, optional
            Whether or not to normalize q and L by rho(x)^2.  Default is True (perform normalization)
        metric : string
            Metric to use for computing distance.  Default is "Euclidean".  See sklearn documentation for more options.
        metric_params : dict
            Additional parameters needed for estimating the metric.
        verbosity : int, optional
            Whether to print verbose output.  If 0 (default), no updates are printed.  If 1, prints results of automated bandwidth and dimensionality routines.  If 2, prints program status updates.

        """
        self.nneighbors = nneighbors
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.epses = epses
        self.eps = None
        self.rho_norm = rho_norm
        self.metric = metric
        self.metric_params = metric_params
        self.verbosity = verbosity

    def fit(self, data, rho=None, point_weights=None):
        """Constructs the diffusion map on the dataset.

        Parameters
        ----------
        data : 2D array-like or dynamical dataset
            Two-dimensional dataset used to create the diffusion map.
        rho : 1d array-like or None, optional
            Bandwidth function to be used in the variable bandwidth kernel.
            If None, the code estimates the density of the data q using a kernel density estimate,
            and sets the bandwidth to q_\epsilon^beta.
        point_weights : 1D array-like or None, optional
            Importance sampling weights for each datapoint.

        """
        # Default Parameter Selection and Type Cleaning
        if len(np.shape(data)) == 1:  # If data is 1D, make it 2D so indices work
            data = np.array([data])
            data = np.transpose(data)
        self.data = data
        d = self.d

        N = len(data)
        d_kde = None  # Density estimate from kde
        if rho is None:  # If no bandwidth fxn given, get one from KDE.
            rho, d_kde = self.get_bandwidth_fxn()
            if self.verbosity >= 1:
                if d_kde is None:
                    print("No Diffusion Map Bandwidth given.  Bandwidth constructed using a KDE.  No dimensionality info detected.")
                else:
                    print("No Diffusion Map Bandwidth given.  Bandwidth constructed using a KDE.  KDE dimension is %d" % d_kde)
        nneighbors = self.nneighbors
        if nneighbors is None:
            nneighbors = N
        nneighbors = np.minimum(nneighbors, N)

        # Evaluate scaled distances
        nn_indices, nn_distsq = get_nns(data, self.nneighbors)
        for i, row in enumerate(nn_distsq):
            row /= rho[i]*rho[nn_indices[i]]

        # Calculate optimal bandwidth
        if isinstance(self.epses, numbers.Number):
            epsilon = self.epses
            if self.verbosity >= 1:
                print("Epsilon provided by the User: %f" % epsilon)
        else:
            epsilon, d_est = self._get_optimal_bandwidth(nn_distsq)
            if self.verbosity >= 1:
                print("Epsilon automatically detected to be : %f" % epsilon)
            if d is None:  # If dimensionality is not provided, use estimated value.
                d = d_est
                if self.verbosity >= 1:
                    print("Dimensionality estimated to be %d." % d)

        # Construct sparse kernel matrix.
        nn_distsq /= epsilon
        nn_distsq = np.exp(-nn_distsq)  # Value of the Kernel fxn for Dmaps
        rows = np.outer(np.arange(N), np.ones(nneighbors))
        K = sps.csr_matrix((nn_distsq.flatten(),
                            (rows.flatten(), nn_indices.flatten())), shape=(N, N))
        if self.verbosity >= 2:
            print("Evaluated Kernel")

        # Symmetrize K using 'or' operator.
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K + dK
        K *= 0.5

        # Apply q^alpha normalization.
        q = np.array(K.sum(axis=1)).flatten()
        if self.rho_norm:
            if np.any(rho-1.):  # Check if bandwidth function is nonuniform.
                if d is None:
                    if d_kde is None:
                        raise ValueError('Dimensionality needed to normalize the density estimate , but no dimensionality information found or estimated.')
                    else:
                        d = d_kde
                q /= (rho**d)
        alpha = _eval_param(self.alpha, d)
        if alpha != 0:
            diagq = sps.dia_matrix((1./(q**alpha), [0]), shape=(N, N))
            K = diagq * K
            K = K * diagq
            if self.verbosity >= 2:
                print(r"Applied q**\alpha normalization.")

        # Apply importance sampling weights if provided.
        if point_weights is not None:
            diag_wt = sps.dia_matrix((point_weights**0.5, [0]), shape=(N, N))
            K = diag_wt * K
            K = K * diag_wt

        # Normalize to Transition Rate Matrix
        q_alpha = np.array(K.sum(axis=1)).flatten()
        diagq_alpha = sps.dia_matrix((1./(q_alpha), [0]), shape=(N, N))
        L = diagq_alpha * K  # Normalize row sum to one.
        diag = L.diagonal()-1.
        L.setdiag(diag)  # subtract identity.
        if self.verbosity >= 2:
            print(r"Applied q**\alpha normalization.")

        # Normalize matrix by epsilon, and (if specified) by bandwidth fxn.
        if self.rho_norm:
            diag_norm = sps.dia_matrix((1./(rho**2*epsilon), 0), shape=(N, N))
        else:
            diag_norm = sps.eye(N)*(1./epsilon)
            pi = q_alpha
        L = diag_norm * L
        if self.verbosity >= 2:
            print("Normalized matrix to transition rate matrix.")

        # Calculate stationary density.
        if self.rho_norm:
            pi = rho**2 * q_alpha
        else:
            pi = q_alpha
        pi /= np.sum(pi)
        if self.verbosity >= 2:
            print("Estimated Stationary Distribution.")

        # Return calculated quantities.
        self.K = K
        self.L = L
        self.pi = L
        self.epsilon = epsilon

    def get_bandwidth_fxn(self):
        N = len(self.data)
        if ((self.beta == 0) or (self.beta == '0')):
            return np.ones(N), None  # Handle uniform bandwidth case.
        else:
            # Use q^beta as bandwidth, where q is an estimate of the density.
            q, d_est, eps_opt = kde(self.data, epses=self.epses, nneighbors=self.nneighbors, d=self.d)
            if self.d is None:
                d = d_est

            # If beta parameter is an expression, evaluate it and convert to float
            print(self.beta, d)
            beta = _eval_param(self.beta, d)
            return q**beta, d

    def _get_optimal_bandwidth(self, scaled_distsq):
        if isinstance(self.epses, numbers.Number):
            return self.epses, self.d
        else:
            return get_optimal_bandwidth_BH(scaled_distsq, self.epses)

    def make_dirichlet_basis(self, k, outside_domain=None):
        """Creates a diffusion map basis set that obeys the homogeneous
        Dirichlet boundary conditions on the domain.  This is done by taking
        the eigenfunctions of the diffusion map submatrix on the domain.

        Parameters
        ----------
        k : int
            Number of basis functions to create.
        outside_domain : 1D array-like, optional
            Array of the same length as the data, where each element is 1 or True if that datapoint is outside the domain, and 0 or False if it is in the domain.  Naturally, this must be the length as the current dataset.  If None (default), all points assumed to be in the domaain.


        Returns
        -------
        basis : 2D numpy array
            The basis functions.
        evals : 1D numpy array
            The eigenvalues corresponding to each basis vector.

        """
        submat = self.L
        if outside_domain is not None:
            domain = 1 - outside_domain
            submat = submat[domain][:, domain]
        evals, evecs = spsl.eigs(submat, k, which='LR')
        # Sort by eigenvalue.
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        return evecs, evals


def kde(data, rho=None, nneighbors=None, d=None, nn_rho=8, epses=2.**np.arange(-40, 41), verbosity=0, metric='euclidean', metric_params=None):
    """Code implementing Kernel Density estimatation.  Algorithm is heavily based on that presented in Berry, Giannakis, and Harlim, Phys. Rev. E. 91, 032915 (2015).

    Parameters
    ----------
    data : ndarray
        Data to create the diffusion map on.  Can either be a one-dimensional time series, or a timeseries of Nxk, where N is the number of data points and k is the dimensionality of data.
    rho : ndarray or None, optional
        Bandwidth function rho(x) to use in the kernel, evaluated at each data point.  The kernel used is exp(-||x-y||^2/(rho(x)rho(y))).  If None is given (default), uses the automatic bandwidth procedure defined using nearest neighbors, as given by BGH.
    nneighbors : int, optional
        Approximate number of nearest neighbors to keep for each state in the kernel matrix.  This introduces a truncation error into the diffusion map.  However, due to exponential decay of the Kernel, this is generally negligible.  Default is None, i.e. to keep all neighbors.
    d : int, optional
        Dimensionality of the data.  If not given, detected automatically as part of the automatic bandwidth detection procedure.  Note that automatic bandwidth detection will only give good results if the values of epsilon provided include the optimal region descriped by Coifman and Lafon and BGH.
    nn_rho : int, optional
        Number of nearest neighbors to use when constructing the automatic bandwidth function.  Default is 8.  If rho is provided by the user, this does nothing.
    epses : array-like, optional
        Epsilon values to be used for automatic bandwidth detection.  Requires at least three values.  Default is powers of 2 between 2^40 and 2^-40.  Note that the optimal epsilon value used will actually be *between* these values, due to it being estimated using a central difference of a function of the epsilon values.
    metric : string
        Metric to use for computing distance.  Default is "Euclidean".  See sklearn documentation for more options.
    metric_params : dict
        Additional parameters needed for estimating the metric.
    verbosity : int, optional
        Whether to print verbose output.  If 0 (default), no updates are printed.  If 1, prints results of automated bandwidth and dimensionality routines.  If 2, prints program status updates.

    Returns
    -------
    q : 1d array
        Estimated value of the Density.
    d_est : int
        Estimated dimensionality of the system.  This is not necessarily the same as the dimensionality used in the calculation if the user provides a value of d.
    eps : float
        Optimal bandwidth parameter for the system.
    """
    # Default Parameter Selection and Type Cleaning
    N = len(data)
    if nneighbors is None:
        nneighbors = N  # If no neighbor no. provided, use full data set.
    if len(np.shape(data)) == 1:  # If data is 1D structure, make it 2D
        data = np.array([data])
        data = np.transpose(data)
    data = np.array([dat for dat in data])

    # Get nearest neighbors
    nn_indices, nn_distsq = get_nns(data, nneighbors)

    # Construct a bandwidth function if none is provided by the user.
    if rho is None:
        rho_indices = np.zeros((N, nn_rho), dtype=np.int)
        rho_distsq = np.zeros((N, nn_rho))
        for i, row in enumerate(nn_distsq):
            # Get nearest nn_rho points to point i
            row_indices = np.argpartition(row, nn_rho-1)[:nn_rho]
            row_d2 = row[row_indices]
            rho_indices[i] = row_indices
            rho_distsq[i] = row_d2
        rho = np.sqrt(np.sum(rho_distsq, axis=1)/(nn_rho-1))

    # Perform automatic bandwidth selection.
    scaled_distsq = np.copy(nn_distsq)
    for i, row in enumerate(scaled_distsq):
        row /= 2.*rho[i]*rho[nn_indices[i]]

    if isinstance(epses, numbers.Number):
        eps_opt = epses
    else:
        eps_opt, d_est = get_optimal_bandwidth_BH(scaled_distsq, epses=epses)
        if d is None:  # If dimensionality is not provided, use estimated value.
            d = d_est

    # Estimated density.
    q0 = np.sum(np.exp(-scaled_distsq/eps_opt), axis=1)
    if np.any(rho-1.):
        if d is None:
            raise ValueError('Dimensionality needed to normalize the density estimate , but no dimensionality information found or estimated.')
    q0 /= (rho**d)
    q0 *= (2.*np.pi)**(-d/2.) / len(q0)
    return q0, d, eps_opt


def get_optimal_bandwidth_BH(scaled_distsq, epses=2.**np.arange(-40, 41)):
    """Calculates the optimal bandwidth for kernel density estimation, according to the algorithm of Berry and Harlim.

    Parameters
    ----------
    scaled_distsq : 1D array-like
        Value of the distances squared, scaled by the bandwidth function.  For instance, this could be ||x-y||^2 / (\rho(x) \rho(y)) evaluated at each pair of points.
    epses : 1D array-like, optional
        Possible values of the bandwidth constant.  The optimal value is selected by estimating the derivative in Giannakis, Berry, and Harlim using a forward difference.  Note: it is explicitely assumed that the the values are either monotonically increasing or decreasing.  Default is all powers of two from 2^-40 to 2^40.
    """
    # Calculate double sum.
    log_T = []
    log_eps = []
    for eps in epses:
        #        kernel = np.exp(-scaled_distsq/float(eps))
        kernel = np.exp(-scaled_distsq/float(eps))
        log_T.append(np.log(np.average(kernel)))
        log_eps.append(np.log(eps))

    # Find max of derivative of d(log(T))/d(log(epsilon)), get optimal eps, d
    log_deriv = np.diff(log_T)/np.diff(log_eps)
    max_loc = np.argmax(log_deriv)
    eps_opt = np.max([np.exp(log_eps[max_loc]), np.exp(log_eps[max_loc+1])])
    d = np.round(2.*log_deriv[max_loc])
    return eps_opt, d


def get_nns(data, nneighbors=None, sort=False, metric='euclidean', metric_params=None):
    """Get the indices of the nneighbors nearest neighbors, and calculate the distance to them.

    Parameters
    ----------
    data : 2D array-like
        The location of every data point in the space
    period : 1D array-like or float, optional
        Period of the coordinate, e.g. 360 for an angle in degrees. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D array-like with each value a scalar or None, each cv has periodicity of that size.
    nneighbors : int or None, optional
        Number of nearest neighbors to calculate.  If None, calculates all nearest neighbor distances.
    sort : bool, optional
        If True, returns the nearest neighbor distances sorted by distance to each point
        metric : string
            Metric to use for computing distance.  Default is "Euclidean".  See sklearn documentation for more options.
        metric_params : dict
            Additional parameters needed for estimating the metric.

    Returns
    -------
    indices : 2D array
        indices of the nearest neighbers.  Element i,j is the j'th nearest neighbor of the i'th data point.
    distsq : 2D array
        Squared distance between points in the neighborlist.  If D is provided, this matrix is weighted.

    """
#    if period is not None: # Periodicity provided.
#        if not hasattr(period,'__getitem__'): # Check if period is scalar
#            period = [period]
#    else:
#        period = [None]*len(data[0])
    npnts = len(data)
    if nneighbors is None:
        nneighbors = npnts
    nneighbors = np.minimum(nneighbors, npnts)
    indices = np.zeros((npnts, nneighbors), dtype=np.int)
    distsq = np.zeros((npnts, nneighbors))
    for i, pnt in enumerate(data):
        dx = pnt - data
        # Enforce periodic boundary conditions.
#        for dim in range(len(pnt)):
#            p = period[dim]
#            if p is not None:
#                dx[:,dim] -= p*np.rint(dx[:,dim]/p)

        dsq_i = np.sum(dx*dx, axis=1)
        # Find nneighbors largest elements
        ui_i = np.argpartition(dsq_i, nneighbors-1)[:nneighbors]  # unsorted indices
        ud_i = dsq_i[ui_i]  # unsorted distances
        if sort:
            sorter = ud_i.argsort()
            indices[i] = ui_i[sorter]
            distsq[i] = ud_i[sorter]
        else:
            indices[i] = ui_i
            distsq[i] = ud_i
    return indices, distsq


def _eval_param(param, d):
    """
    Evaluates the alpha or beta parameters.  For instance, if the user passes "1/d", this must be converted to a float.

    Parameters
    ----------
    param : float or string
        The parameter to be evaluated.  Either a float or an evaluatable string, where "d" stands in for the system dimensionality.
    d : int
        Dimensionality of the system.

    Returns
    -------
    eval_param : float
        The value of the parameter, evaluated.
    """
    nsp = _NumericStringParser()
    if type(param) is str:
        if 'd' in param:
            if d is None:
                raise ValueError('Dimensionality needed in evaluating %s, but no dimensionality information found or estimated.' % param)
            param = param.replace('d', str(d))
        print(param)
        return nsp.eval(param)
    else:
        return float(param)
