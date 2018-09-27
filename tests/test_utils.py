import numpy as np
import pytest

from pydiffmap import utils
from sklearn.neighbors import NearestNeighbors

x_1d = np.arange(10)
x_2d = np.arange(20).reshape(10, 2)
y_1d = np.arange(10) + 0.5
y_2d = np.arange(20).reshape(10, 2) + 0.5


class TestLookupFunction(object):
    @pytest.mark.parametrize('x', [x_1d, x_2d])
    @pytest.mark.parametrize('vals', [y_1d, y_2d])
    def test_lookup_fxn(self, x, vals):
        N = len(x)
        shuffle_indices = np.arange(N)
        np.random.shuffle(shuffle_indices)
        lf = utils.lookup_fxn(x, vals)
        shuffle_y = np.array([lf(xi) for xi in x[shuffle_indices]])
        assert((shuffle_y == vals[shuffle_indices]).all())


class TestSparseFromFxn(object):
    @pytest.mark.parametrize('Y', [y_2d, None])
    def test_sparse_from_fxn(self, Y):
        nneighbors = NearestNeighbors(10)
        nneighbors.fit(x_2d)
        Y2 = Y
        if Y2 is None:
            Y2 = x_2d
        ref_mat = nneighbors.kneighbors_graph(Y2, mode='distance')
        dist_fxn = lambda Y, X: np.linalg.norm(Y - X)
        dist_mat = utils.sparse_from_fxn(nneighbors, dist_fxn, Y)
        assert(np.linalg.norm((dist_mat - ref_mat).data) < 1e-10)
