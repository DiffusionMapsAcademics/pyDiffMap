import matplotlib.pyplot as plt
import numpy as np
import pytest

from pydiffmap import diffusion_map as dm
from pydiffmap import visualization as viz


@pytest.fixture(scope='module')
def dummy_dmap(uniform_2d_data):
    data, X, Y = uniform_2d_data
    print(data)
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=3, k=5)
    mydmap.fit(data)
    return mydmap


class TestEmbeddingPlot():
    @pytest.mark.parametrize('dim', [2, 3])
    def test_no_kwargs(self, dummy_dmap, dim):
        mydmap = dummy_dmap
        fig = viz.embedding_plot(mydmap, dim=dim, scatter_kwargs=None, show=False)
        assert(fig)

    def test_fixed_coloring(self, dummy_dmap):
        mydmap = dummy_dmap
        scatter_kwargs = {'c': 'r'}
        true_coloring = (1.0, 0., 0., 1)
        fig = viz.embedding_plot(mydmap, scatter_kwargs=scatter_kwargs, show=False)
        SC = fig.axes[0].collections[0]
        assert(np.all(SC._facecolors[0] == true_coloring))

    @pytest.mark.parametrize('size', [4., np.arange(1, 82)])
    def test_size(self, dummy_dmap, size):
        mydmap = dummy_dmap
        scatter_kwargs = {'s': size}
        fig = viz.embedding_plot(mydmap, scatter_kwargs=scatter_kwargs, show=False)
        SC = fig.axes[0].collections[0]
        actual_sizes = SC.get_sizes()
        assert(np.all(actual_sizes == size))

    @pytest.mark.parametrize('cmap', [None, 'Blues', plt.cm.Spectral])
    def test_colormap(self, dummy_dmap, cmap):
        # This just tests if the code runs...
        # Replace with something more stringent?
        mydmap = dummy_dmap
        scatter_kwargs = {'c': mydmap.dmap[:, 0], 'cmap': cmap}
        fig = viz.embedding_plot(mydmap, scatter_kwargs=scatter_kwargs, show=False)
        assert(fig)


class TestDataPlot():
    def test_no_kwargs(self, dummy_dmap):
        mydmap = dummy_dmap
        fig = viz.data_plot(mydmap, scatter_kwargs=None, show=False)
        assert(fig)

    @pytest.mark.parametrize('size', [4., np.arange(1, 82)])
    def test_size(self, dummy_dmap, size):
        mydmap = dummy_dmap
        scatter_kwargs = {'s': size}
        fig = viz.data_plot(mydmap, 1, scatter_kwargs=scatter_kwargs, show=False)
        SC = fig.axes[0].collections[0]
        actual_sizes = SC.get_sizes()
        assert(np.all(actual_sizes == size))
