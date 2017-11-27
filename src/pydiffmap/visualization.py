# -*- coding: utf-8 -*-
"""
Some convenient visalisation routines.
"""
from __future__ import absolute_import

import matplotlib.pyplot as plt


def embedding_plot(dmap_instance, scatter_kwargs=None, show=True):
    """
    Creates diffusion map embedding scatterplot. By default, the first two diffusion
    coordinates are plotted against each other.

    Parameters
    ----------
    dmap_instance : DiffusionMap Instance
        An instance of the DiffusionMap class.
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
    plt.scatter(dmap_instance.dmap[:, 0], dmap_instance.dmap[:, 1], **scatter_kwargs)
    plt.title('Embedding')
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.axis('tight')
    if show:
        plt.show()
    return fig

def data_plot(dmap_instance, n_evec=1, scatter_kwargs=None, show=True):
    """
    Creates diffusion map embedding scatterplot. By default, the first two diffusion
    coordinates are plotted against each other.  This only plots against the first two
    dimensions of the data, however: effectively this assumes the data is two dimensional.

    Parameters
    ----------
    dmap_instance : DiffusionMap Instance
        An instance of the DiffusionMap class.
    n_evec: int, optional
        The eigenfunction that should be used to color the plot.
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
    fig =  plt.figure(figsize=(6,6))
    plt.scatter(dmap_instance.data[:, 0], dmap_instance.data[:, 1], c=dmap_instance.dmap[:,n_evec-1], **scatter_kwargs)
    plt.title('Data coloured with first DC.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    if show:
        plt.show()
    return fig
