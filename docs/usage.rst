=====
Usage
=====

To use pyDiffMap in a project::

	import pyDiffMap

To initialize a diffusion map object::

	mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = 1, epsilon = 1.0, alpha = 0.5, k=64)

where ``n_evecs`` is the number of eigenvectors that are computed, ``epsilon`` is a scale parameter
used to rescale distances between data points, ``alpha`` is a normalization parameter (typically between 0.0 and 1.0)
that influences the effect of the sampling density, and ``k`` is the number of nearest neighbors considered when the kernel
is computed. A larger ``k`` means increased accuracy but larger computation time. 
The ``from_sklearn`` command is used because we are constructing using the scikit-learn nearest neighbor framework.
For additional optional arguments, see documentation.

We can also employ automatic epsilon detection due to an algorithm by Berry, Harlim and Giannakis::

	mydmap = dm.DiffusionMap.from_sklearn(n_evecs = 1, alpha = 0.5, epsilon = 'bgh', k=64)

To fit to a dataset ``X`` (array-like, shape (n_query, n_features))::

	mydmap.fit(X)

The diffusion map coordinates can also be accessed directly via::

	dmap = mydmap.fit_transform(X)

This returns an array ``dmap`` with shape (n_query, n_evecs). E.g. ``dmap[:,0]`` is the first diffusion coordinate
evaluated on the data ``X``.

In order to compute diffusion coordinates at the out of sample location(s) ``Y``::

	dmap_Y = mydmap.transform(Y)
