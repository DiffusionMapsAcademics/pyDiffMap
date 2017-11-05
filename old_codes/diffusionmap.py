"""Diffusion map"""

# Author: Zofia
# License:


import numpy as np

import math
from scipy.sparse import csr_matrix

import scipy.sparse.linalg as SLA

import sklearn.neighbors as neigh_search
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps

import mdtraj as md
import rmsd

from numpy import linalg as LA

import model


dummyModel=model.dummyModel
#


def compute_kernel(X, epsilon):
    """
    Compute kernel of trajectory: using RMSD distances
    parameters: X is matrix of size number of steps times nDOF
    """


    m = np.shape(X)[0];

    cutoff = np.sqrt(2*epsilon);

    #calling nearest neighbor search class: returning a (sparse) distance matrix
    #albero = neigh_search.radius_neighbors_graph(X, radius = cutoff, mode='distance', p=2, include_self=None)
    print('constructing neighbors graph')
    albero = neigh_search.radius_neighbors_graph(X, radius=cutoff, mode='distance', metric = min_rmsd, include_self=None)#mode='pyfunc',, metric_params={'myRMSDmetric':myRMSDmetric}, include_self=None)
    print('neighbors graph done')
    #albero = neigh_search.radius_neighbors_graph(X.xyz, radius=cutoff, mode='pyfunc', metric_params={'func' : md.rmsd}, include_self=None)

    #adaptive epsilon
    x=np.array(albero.data)
    adaptiveEpsilon=0.5*np.mean(x)
    diffusion_kernel = np.exp(-(x**2)/(adaptiveEpsilon))
    #print("Adaptive epsilon in compute_kernel is "+repr(adaptiveEpsilon))

    # adaptive epsilon should be smaller as the epsilon, since it is half of maximal distance which is bounded by cutoff parameter
    assert( adaptiveEpsilon <= epsilon )

    # computing the diffusion kernel value at the non zero matrix entries
    #diffusion_kernel = np.exp(-(np.array(albero.data)**2)/(epsilon))

    # build sparse matrix for diffusion kernel
    kernel = sps.csr_matrix((diffusion_kernel, albero.indices, albero.indptr), dtype = float, shape=(m,m))
    kernel = kernel + sps.identity(m)  # accounting for diagonal elements

    return kernel;

def myEuclideanMetric(arr1, arr2):
    """
    Under assumption that the trajectory is aligned w.r.t minimal rmsd w.r.t. first frame
    This is built under the assumption that the space dimension is 3!!!
    Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
     Input: One row from reshaped xyz trajectory as number of steps times nDOF
     Inside: Reshape back to xyz (NrPart, dim) and apply norm
     Output: r
    """


    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    arr1 = arr1.reshape(int(nParticles), 3 )
    arr2 = arr2.reshape(int(nParticles), 3 )

    s=0
    for i in range(int(nParticles)):
        stmp = np.linalg.norm(arr1[i,:]-arr2[i,:])
        s+=stmp*stmp
    d=np.sqrt( s / nParticles)


    return d


def myRMSDmetric(arr1, arr2):
    """
    This is built under the assumption that the space dimension is 3!!!
    Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
     Input: One row from reshaped xyz trajectory as number of steps times nDOF
     Inside: Reshape back to md.Trajectory and apply md.rmsd as r=md.rmsd(X[i], X[j])
     Output: r
    """

    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    arr1 = arr1.reshape(int(nParticles), 3 )
    arr2 = arr2.reshape(int(nParticles), 3 )


    p1MD=md.Trajectory(arr1, dummyModel.testsystem.topology)
    p2MD=md.Trajectory(arr2, dummyModel.testsystem.topology)

    d=md.rmsd(p1MD, p2MD)

    return d

def min_rmsd(arr1, arr2):

    nParticles = len(arr1) / 3;
    assert (nParticles == int(nParticles))

    X1 = arr1.reshape(int(nParticles), 3 )
    X2 = arr2.reshape(int(nParticles), 3 )

    X1 = X1 -  rmsd.centroid(X1)
    X2 = X2 -  rmsd.centroid(X2)

    x = rmsd.kabsch_rmsd(X1, X2)

    return x




def compute_P(kernel, X):
    """
    VANILLA diffusion map
    """

    alpha = 0.5;
    m = np.shape(X)[0];
    D = sps.csr_matrix.sum(kernel, axis=0);
    Dalpha = sps.spdiags(np.power(D,-alpha), 0, m, m)
    kernel = Dalpha * kernel * Dalpha;
    D = sps.csr_matrix.sum(kernel, axis=0);
    Dalpha = sps.spdiags(np.power(D,-1), 0, m, m)
    kernel = Dalpha * kernel;

    return kernel
