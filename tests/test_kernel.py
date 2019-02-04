import numpy as np
import pytest

from pydiffmap import kernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

x_values = np.vstack((np.linspace(-1, 1, 11), np.arange(11))).T  # set of X vals
y_values_set = [None, x_values, np.arange(6).reshape(-1, 2), np.arange(22).reshape(-1, 2)]  # all sets of Y's
bandwidth_fxns = [None, lambda x: np.ones(x.shape[0]), lambda x: x[:, 1]/10. + 1.]
epsilons = [10., 1.]  # Possible epsilons


class TestKernel(object):
    # These decorators run the test against all possible y, epsilon values.
    @pytest.mark.parametrize('y_values', y_values_set)
    @pytest.mark.parametrize('epsilon', epsilons)
    @pytest.mark.parametrize('bandwidth_fxn', bandwidth_fxns)
    @pytest.mark.parametrize('metric, metric_params', [
        ('euclidean', None),
        ('minkowski', {'p': 1})
    ])
    def test_matrix_output(self, y_values, epsilon, bandwidth_fxn, metric, metric_params):
        """
        Test that we are returning the correct kernel values.
        """
        # Setup true values to test again.
        if y_values is None:
            y_values_ref = x_values
        else:
            y_values_ref = y_values
        if metric == 'minkowski':
            pw_distance = cdist(y_values_ref, x_values, metric='minkowski', p=metric_params['p'])
        else:
            pw_distance = cdist(y_values_ref, x_values, metric=metric)
        if bandwidth_fxn is None:
            ref_bandwidth_fxn = lambda x: np.ones(x.shape[0])
        else:
            ref_bandwidth_fxn = bandwidth_fxn
        x_bandwidth = ref_bandwidth_fxn(x_values)
        y_bandwidth = ref_bandwidth_fxn(y_values_ref).reshape(-1, 1)
        scaled_sq_dists = pw_distance**2 / (x_bandwidth * y_bandwidth)
        true_values = np.exp(-1.*scaled_sq_dists/(4. * epsilon))

        # Construct the kernel and fit to data.
        mykernel = kernel.Kernel(kernel_type='gaussian', metric=metric,
                                 metric_params=metric_params, epsilon=epsilon,
                                 k=len(x_values), bandwidth_type=bandwidth_fxn)
        mykernel.fit(x_values)
        K_matrix = mykernel.compute(y_values).toarray()

        # Check that error values are beneath tolerance.
        error_values = (K_matrix-true_values).ravel()
        total_error = np.linalg.norm(error_values)
        assert(total_error < 1E-8)

    @pytest.mark.parametrize('k', np.arange(2, 14, 2))
    @pytest.mark.parametrize('neighbor_params', [{'algorithm': 'auto'}, {'algorithm': 'ball_tree'}])
    def test_neighborlists(self, k, neighbor_params):
        """
        Test that neighborlisting gives the right number of elements.
        """
        # Correct number of nearest neighbors.
        k0 = min(k, len(x_values))

        # Construct kernel matrix.
        mykernel = kernel.Kernel(kernel_type='gaussian', metric='euclidean',
                                 epsilon=1., k=k0, neighbor_params=neighbor_params)
        mykernel.fit(x_values)
        K_matrix = mykernel.compute(x_values)

        # Check if each row has correct number of elements
        row_has_k_elements = (K_matrix.nnz == k0*len(x_values))
        assert(row_has_k_elements)

    @pytest.mark.parametrize('eps_method', ['bgh', 'bgh_generous'])
    def test_auto_epsilon_selection(self, eps_method):
        X = np.arange(100).reshape(-1, 1)
        mykernel = kernel.Kernel(kernel_type='gaussian', metric='euclidean',
                                 epsilon=eps_method, k=10)
        mykernel.fit(X)
        if eps_method == 'bgh':
            assert(mykernel.epsilon_fitted == 0.25)
        else:
            assert(mykernel.epsilon_fitted == 0.50)
        assert(mykernel.d == 1.0)


class TestKNN(object):
    def test_harmonic_kde(self, harmonic_1d_data):
        # Setup Data
        data = harmonic_1d_data
        Y = np.linspace(-2.5, 2.5, 201)
        oos_data = Y.reshape(-1, 1)
        ref_density = np.exp(-Y**2 / 2.) / np.sqrt(2 * np.pi)
        THRESH = 0.003
        # Build kde object
        nneighbs = NearestNeighbors(n_neighbors=120)
        nneighbs.fit(data)
        my_kde = kernel.NNKDE(nneighbs, k=16)
        my_kde.fit()
        density = my_kde.compute(oos_data)
        error = np.sqrt(np.mean((density - ref_density)**2))
        assert(error < THRESH)


class TestBGHEpsilonSelection(object):
    @pytest.mark.parametrize('k', [10, 30, 100])
    def test_1D_uniform_data(self, k):
        X = np.arange(100).reshape(-1, 1)
        neigh = NearestNeighbors(n_neighbors=k)
        sq_dist = neigh.fit(X).kneighbors_graph(X, mode='distance').data**2.
        epsilons = 2**np.arange(-20., 20.)
        eps, d = kernel.choose_optimal_epsilon_BGH(sq_dist, epsilons)
        assert(eps == 0.25)
        assert(d == 1.0)
