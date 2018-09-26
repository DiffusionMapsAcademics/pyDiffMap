import numpy as np
import pytest

from pydiffmap import utils


class TestLookupFunction(object):
    x_1d = np.arange(10)
    x_2d = np.arange(20).reshape(10, 2)
    y_1d = np.arange(10)**2
    y_2d = np.arange(20).reshape(10, 2)**2

    @pytest.mark.parametrize('x', [x_1d, x_2d])
    @pytest.mark.parametrize('vals', [y_1d, y_2d])
    def test_lookup_fxn(self, x, vals):
        N = len(x)
        shuffle_indices = np.arange(N)
        np.random.shuffle(shuffle_indices)
        lf = utils.lookup_fxn(x, vals)
        shuffle_y = np.array([lf(xi) for xi in x[shuffle_indices]])
        assert((shuffle_y == vals[shuffle_indices]).all())
