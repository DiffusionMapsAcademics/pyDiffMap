import numpy as np
import pytest


@pytest.fixture(scope='module')
def spherical_data():
    # Construct dataset
    phi = np.pi*np.linspace(-1, 1, 61)[1:]
    theta = np.pi*np.linspace(-1, 1, 33)[1:-1]
    Phi, Theta = np.meshgrid(phi, theta)
    Phi = Phi.ravel()
    Theta = Theta.ravel()

    X = np.cos(Theta)*np.cos(Phi)
    Y = np.cos(Theta)*np.sin(Phi)
    Z = np.sin(Theta)
    return np.array([X, Y, Z]).transpose(), Phi, Theta


@pytest.fixture(scope='module')
def uniform_2d_data():
    x = np.linspace(0., 1., 61)*2.*np.pi
    y = np.linspace(0., 1., 31)*np.pi
    X, Y = np.meshgrid(x, y)
    X = X.ravel()
    Y = Y.ravel()
    data = np.array([X, Y]).transpose()
    return data, X, Y
