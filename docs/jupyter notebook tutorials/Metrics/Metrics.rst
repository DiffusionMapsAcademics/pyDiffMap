
Diffusion maps with general metric
==================================

In this notebook, we illustrate how to use an optional metric in the
diffusion maps embedding.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from mpl_toolkits.mplot3d import Axes3D
    from pydiffmap import diffusion_map as dm
    
    %matplotlib inline

We import trajectory of two particles connected by a double-well
potential, which is a function of a radius: V(r) = V\_DW(r). The dimer
was simulated at 300K with Langevin dynamics using OpenMM. The obvious
collective variable is the radius case and we demonstrate how the first
dominant eigenvector obtained from the diffusion map clearly correlates
with this reaction coordinate. As a metric, we use the root mean square
deviation (RMSD) from the package
https://pypi.python.org/pypi/rmsd/1.2.5.

.. code:: ipython3

    traj=np.load('Data/dimer_trajectory.npy')
    energy=np.load('Data/dimer_energy.npy')
    print('Loaded trajectory of '+repr(len(traj))+' steps of dimer molecule: '+repr(traj.shape[1])+' particles in dimension '+repr(traj.shape[2])+'.')


.. parsed-literal::

    Loaded trajectory of 1000 steps of dimer molecule: 2 particles in dimension 3.


.. code:: ipython3

    def compute_radius(X):
        return np.linalg.norm(X[:,0,:]-X[:,1,:], 2, axis=1)
    
    fig = plt.figure(figsize=[16,6])
    ax = fig.add_subplot(121)
    
    radius= compute_radius(traj)
    cax2 = ax.scatter(range(len(radius)), radius, c=radius, s=20,alpha=0.90,cmap=plt.cm.Spectral)
    cbar = fig.colorbar(cax2)
    cbar.set_label('Radius')
    ax.set_xlabel('Simulation steps')
    ax.set_ylabel('Radius')
    
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    L=2
    
    i=0
    
    ax2.scatter(traj[i,0,0], traj[i,0,1], traj[i,0,2], c='b', s=100, alpha=0.90, edgecolors='none', depthshade=True,)
    ax2.scatter(traj[i,1,0], traj[i,1,1], traj[i,1,2], c='r', s=100, alpha=0.90, edgecolors='none',  depthshade=True,)
        
    ax2.set_xlim([-L, L])
    ax2.set_ylim([-L, L])
    ax2.set_zlim([-L, L])
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
       
    plt.show()




.. image:: output_5_0.png


.. code:: ipython3

    # download from https://pypi.python.org/pypi/rmsd/1.2.5
    import rmsd
    
    
    def myRMSDmetric(arr1, arr2):
        """
        This function is built under the assumption that the space dimension is 3!!!
        Requirement from sklearn radius_neighbors_graph: The callable should take two arrays as input and return one value indicating the distance between them.
         Input: One row from reshaped XYZ trajectory as number of steps times nDOF
         Inside: Reshape to XYZ format and apply rmsd as r=rmsd(X[i], X[j])
         Output: rmsd distance
        """
        
        nParticles = len(arr1) / 3;
        assert (nParticles == int(nParticles))
    
        X1 = arr1.reshape(int(nParticles), 3 )
        X2 = arr2.reshape(int(nParticles), 3 )
    
        X1 = X1 -  rmsd.centroid(X1)
        X2 = X2 -  rmsd.centroid(X2)
    
        return rmsd.kabsch_rmsd(X1, X2)
    


Compute diffusion map embedding using the rmsd metric from above.

.. code:: ipython3

    epsilon=0.05
    
    Xresh=traj.reshape(traj.shape[0], traj.shape[1]*traj.shape[2])
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs = 1, epsilon = epsilon, alpha = 0.5, k=1000, metric=myRMSDmetric)
    dmap = mydmap.fit_transform(Xresh)


.. parsed-literal::

    0.05 eps fitted


Plot the dominant eigenvector over radius, to show the correlation with
this collective variable.

.. code:: ipython3

    evecs = mydmap.evecs
    
    fig = plt.figure(figsize=[16,6])
    ax = fig.add_subplot(121)
    
    ax.scatter(compute_radius(traj), evecs[:,0], c=evecs[:,0], s=10, cmap=plt.cm.Spectral)
    ax.set_xlabel('Radius')
    ax.set_ylabel('Dominant eigenvector')
    
    ax2 = fig.add_subplot(122)
    #
    cax2 = ax2.scatter(compute_radius(traj), energy, c=evecs[:,0], s=10, cmap=plt.cm.Spectral)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Potential Energy')
    cbar = fig.colorbar(cax2)
    cbar.set_label('Dominant eigenvector')
    plt.show()



.. image:: output_10_0.png


