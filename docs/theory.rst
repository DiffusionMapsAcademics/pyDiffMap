======
Theory
======

Diffusion maps is a dimension reduction technique that can be used to discover low dimensional structure in high
dimensional data. It assumes that the data points, which are given as points in a high dimensional metric space,
actually live on a lower dimensional structure. To uncover this structure, diffusion maps builds a neighborhood graph
on the data based on the distances between nearby points. Then a graph Laplacian **L** is constructed on the neighborhood
graph. Many variants exist that approximate different differential operators. For example, *standard* diffusion maps
approximates the differential operator

.. math::

   \mathcal{L}f = \Delta f - 2(1-\alpha)\nabla f \cdot \frac{\nabla q}{q}


where :math:`\Delta` is the Laplace Beltrami operator, :math:`\nabla` is the gradient operator and :math:`q` is the
sampling density. The normalization parameter :math:`\alpha`, which is typically between 0.0 and 1.0, determines how
much :math:`q` is allowed to bias the operator :math:`\mathcal{L}`.
Standard diffusion maps on a dataset ``X``, which has to given as a numpy array with different rows corresponding to
different observations, is implemented in pydiffmap as::

  mydmap = diffusion_map.DiffusionMap.from_sklearn(epsilon = my_epsilon, alpha = my_alpha)
  mydmap.fit(X)

Here ``epsilon`` is a scale parameter used to rescale distances between data points. 
We can also choose ``epsilon`` automatically due to an an algorithm by Berry, Harlim and Giannakis::

  mydmap = dm.DiffusionMap.from_sklearn(alpha = my_alpha, epsilon = 'bgh')

For additional optional arguments of the DiffusionMap class, see usage and documentation.

A variant of diffusion maps, 'TMDmap', unbiases with respect to :math:`q` and approximates the differential operator

.. math::

  \mathcal{L}f = \Delta f + \nabla (\log\pi) \cdot \nabla f

where :math:`\pi` is a 'target distribution' that defines the drift term and has to be known up to a normalization
constant. TMDmap is implemented in pydiffmap as::

  mydmap = diffusion_map.TMDmap(epsilon = my_epsilon, alpha = 1.0, change_of_measure=com_fxn)
  mydmap.fit(X)

where ``com_fxn`` is function that takes in a coordinate and outputs the value of the target distribution :math:`\pi` .
