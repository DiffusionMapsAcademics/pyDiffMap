# -*- coding: utf-8 -*-
"""
Some convenient visalisation routines.
"""
from __future__ import absolute_import

import matplotlib.pyplot as plt


def embedding_plot(DiffusionMapObject):
    plt.figure(figsize=(6,6))
    plt.scatter(DiffusionMapObject.dmap[:,0],DiffusionMapObject.dmap[:,1], c=DiffusionMapObject.dmap[:,0], cmap=plt.cm.Spectral)
    plt.title('Embedding of Swiss Roll')
    plt.xlabel(r'$\psi_1$')
    plt.ylabel(r'$\psi_2$')
    plt.axis('tight')
    plt.show()
