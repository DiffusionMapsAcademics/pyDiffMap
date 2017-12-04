# -*- coding: utf-8 -*-
"""
Some convenient visalisation routines.
"""
from __future__ import absolute_import

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa F401


def embedding_plot(dmap_instance, dim=2, scatter_kwargs=None, show=True):
    """
    Creates diffusion map embedding scatterplot. By default, the first two diffusion
    coordinates are plotted against each other.

    Parameters
    ----------
    dmap_instance : DiffusionMap Instance
        An instance of the DiffusionMap class.
    dim: int, optional, 2 or 3.
        Optional argument that controls if a two- or three dimensional plot is produced.
    scatter_kwargs : dict, optional
        Optional arguments to be passed to the scatter plot, e.g. point color,
        point size, colormap, etc.
    show : boolean, optional
        If true, calls plt.show()

    Returns
    -------
    fig : pyplot figure object
        Figure object where everything is plotted on.

    Examples
    --------
    # Plots the top two diffusion coords, colored by the first coord.
    >>> scatter_kwargs = {'s': 2, 'c': mydmap.dmap[:,0], 'cmap': 'viridis'}
    >>> embedding_plot(mydmap, scatter_kwargs)

    """
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig = plt.figure(figsize=(6, 6))
    if (dim == 2):
        plt.scatter(dmap_instance.dmap[:, 0], dmap_instance.dmap[:, 1], **scatter_kwargs)
        plt.title('Embedding given by first two DCs.')
        plt.xlabel(r'$\psi_1$')
        plt.ylabel(r'$\psi_2$')
    elif (dim == 3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dmap_instance.dmap[:, 0], dmap_instance.dmap[:, 1], dmap_instance.dmap[:, 2], **scatter_kwargs)
        ax.set_title('Embedding given by first three DCs.')
        ax.set_xlabel(r'$\psi_1$')
        ax.set_ylabel(r'$\psi_2$')
        ax.set_zlabel(r'$\psi_3$')
    plt.axis('tight')
    if show:
        plt.show()
    return fig


def data_plot(dmap_instance, n_evec=1, dim=2, scatter_kwargs=None, show=True):
    """
    Creates diffusion map embedding scatterplot. By default, the first two diffusion
    coordinates are plotted against each other.  This only plots against the first two or three
    (as controlled by 'dim' parameter) dimensions of the data, however:
    effectively this assumes the data is two resp. three dimensional.

    Parameters
    ----------
    dmap_instance : DiffusionMap Instance
        An instance of the DiffusionMap class.
    n_evec: int, optional
        The eigenfunction that should be used to color the plot.
    dim: int, optional, 2 or 3.
        Optional argument that controls if a two- or three dimensional plot is produced.
    scatter_kwargs : dict, optional
        Optional arguments to be passed to the scatter plot, e.g. point color,
        point size, colormap, etc.
    show : boolean, optional
        If true, calls plt.show()

    Returns
    -------
    fig : pyplot figure object
        Figure object where everything is plotted on.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig = plt.figure(figsize=(6, 6))
    if (dim == 2):
        plt.scatter(dmap_instance.data[:, 0], dmap_instance.data[:, 1], c=dmap_instance.dmap[:, n_evec-1], **scatter_kwargs)
        plt.title('Data coloured with first DC.')
        plt.xlabel('x')
        plt.ylabel('y')
    elif (dim == 3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dmap_instance.data[:, 0], dmap_instance.data[:, 1], dmap_instance.data[:, 2], c=dmap_instance.dmap[:, n_evec-1], **scatter_kwargs)
        ax.set_title('Data coloured with first DC.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.axis('tight')
    if show:
        plt.show()
    return fig
