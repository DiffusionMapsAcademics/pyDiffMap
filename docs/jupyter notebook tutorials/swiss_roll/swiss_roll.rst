
Illustration of the diffusion_map class on the classic swiss roll data set
==========================================================================

author: Ralf Banisch

We demonstrate the usage of the diffusion_map class on a two-dimensional
manifold embedded in :math:`\mathbb{R}^3`.

.. code:: python

    # import some necessary functions for plotting as well as the diffusion_map class from pydiffmap.
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    from mpl_toolkits.mplot3d import Axes3D
    from pydiffmap import diffusion_map as dm
    
    %matplotlib inline

Create Data
-----------

We create the dataset: A noisy sampling of the twodimensional “swiss
roll” embedded in :math:`\mathbb{R}^3`. The sampling is such that the
density of samples decreases with the distance from the origin
(non-uniform sampling).

In order to be handled correctly by the diffusion_map class, we must
ensure the data is a numpy array of shape (n_points, n_features).

.. code:: python

    # set parameters
    length_phi = 15   #length of swiss roll in angular direction
    length_Z = 15     #length of swiss roll in z direction
    sigma = 0.1       #noise strength
    m = 10000         #number of samples
    
    # create dataset
    phi = length_phi*np.random.rand(m)
    xi = np.random.rand(m)
    Z = length_Z*np.random.rand(m)
    X = 1./6*(phi + sigma*xi)*np.sin(phi)
    Y = 1./6*(phi + sigma*xi)*np.cos(phi)
    
    swiss_roll = np.array([X, Y, Z]).transpose()
    
    # check that we have the right shape
    print(swiss_roll.shape)


.. parsed-literal::

    (10000, 3)


Run pydiffmap
-------------

Now we initialize the diffusion map object and fit it to the dataset.
Since we are interested in only the first two diffusion coordinates we
set n_evecs = 2, and since we want to unbias with respect to the
non-uniform sampling density we set alpha = 1.0. The epsilon parameter
controls the scale, we tune it automatically with the bgh algorithm. The
k parameter controls the neighbour lists, a smaller k will increase
performance but decrease accuracy.

.. code:: python

    # initialize Diffusion map object.
    mydmap = dm.DiffusionMap(n_evecs = 2, alpha = 1.0, choose_eps = 'bgh', k=200)
    # fit to data and return the diffusion map.
    dmap = mydmap.fit_transform(swiss_roll)

Let’s check which value of epsilon was chosen by the bgh algorithm:

.. code:: python

    print(mydmap.epsilon)


.. parsed-literal::

    0.0625


Visualization
-------------

We show the original data set on the right, with points colored
according to the first diffusion coordinate. On the left, we show the
diffusion map embedding given by the first two diffusion coordinates.
Points are again colored according to the first diffusion coordinate,
which seems to parameterize the :math:`\phi` direction. We can see that
the diffusion map embedding ‘unwinds’ the swiss roll.

.. code:: python

    from pydiffmap.visualization import embedding_plot
    
    plt.figure(figsize=(16,6))
    ax = plt.subplot(121)
    ax.scatter(dmap[:,0],dmap[:,1], c=dmap[:,0], cmap=plt.cm.Spectral)
    ax.set_title('Embedding of Swiss Roll')
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_2$')
    ax.axis('tight')
    
    ax2 = plt.subplot(122,projection='3d')
    ax2.scatter(X,Y,Z, c=dmap[:,0], cmap=plt.cm.Spectral)
    ax2.view_init(75, 10)
    ax2.set_title('swiss roll dataset, color according to $\psi_1$')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.show()



.. image:: output_9_0.png


.. code:: python

    #from pydiffmap.visualization import embedding_plot
    #embedding_plot(mydmap, color=phi, size=mydmap.q)

To get a bit more information out of the embedding, we can scale the
points according to the numerical estimate of the sampling density
(mydmap.q), and color them according to their location in the phi
direction. For comparison, we color the original data set according to
:math:`\phi` this time.

.. code:: python

    plt.figure(figsize=(16,6))
    ax = plt.subplot(121)
    ax.scatter(dmap[:,0],dmap[:,1], s=mydmap.q, c=phi, cmap=plt.cm.Spectral)
    ax.set_title('Embedding of Swiss Roll')
    ax.set_xlabel(r'$\psi_1$')
    ax.set_ylabel(r'$\psi_2$')
    ax.axis('tight')
    
    ax2 = plt.subplot(122,projection='3d')
    ax2.scatter(X,Y,Z, c=phi, cmap=plt.cm.Spectral)
    ax2.view_init(75, 10)
    ax2.set_title('swiss roll dataset, color according to $\phi$')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.show()



.. image:: output_12_0.png


We can see that points near the center of the swiss roll, where the
winding is tight, are closer together in the embedding, while points
further away from the center are more spaced out. Let’s check how the
first two diffusion coordinates correlate with :math:`\phi` and
:math:`Z`.

.. code:: python

    print('Correlation between \phi and \psi_1')
    print(np.corrcoef(dmap[:,0], phi))
    
    plt.figure(figsize=(16,6))
    ax = plt.subplot(121)
    ax.scatter(phi, dmap[:,0])
    ax.set_title('First DC against $\phi$')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi_1$')
    ax.axis('tight')
    
    print('Correlation between Z and \psi_2')
    print(np.corrcoef(dmap[:,1], Z))
    
    ax2 = plt.subplot(122)
    ax2.scatter(Z, dmap[:,1])
    ax2.set_title('Second DC against Z')
    ax2.set_xlabel('Z')
    ax2.set_ylabel(r'$\psi_2$')
    
    plt.show()


.. parsed-literal::

    Correlation between \phi and \psi_1
    [[ 1.         -0.92350082]
     [-0.92350082  1.        ]]
    Correlation between Z and \psi_2
    [[ 1.         -0.98102184]
     [-0.98102184  1.        ]]



.. image:: output_14_1.png


