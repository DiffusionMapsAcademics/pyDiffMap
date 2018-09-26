# -*- coding: utf-8 -*-
"""
Utilities for constructing diffusion maps.
"""


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
    def lf(val):
        return lookup[val]

    return lf
