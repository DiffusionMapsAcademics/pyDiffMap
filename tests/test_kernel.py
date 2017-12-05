import numpy as np
import pytest

from pydiffmap import kernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

x_values = np.vstack((np.linspace(-1, 1, 11), np.arange(11))).T  # set of X vals
y_values_set = [None, x_values, np.arange(6).reshape(-1, 2), np.arange(22).reshape(-1, 2)]  # all sets of Y's
epsilons = [10., 1.]  # Possible epsilons


class TestKernel(object):
    # These decorators run the test against all possible y, epsilon values.
    @pytest.mark.parametrize('y_values', y_values_set)
    @pytest.mark.parametrize('epsilon', epsilons)
    @pytest.mark.parametrize('metric, metric_params', [
        ('euclidean', None),
        ('minkowski', {'p': 1})
    ])
    def test_matrix_output(self, y_values, epsilon, metric, metric_params):
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
        true_values = np.exp(-1.*pw_distance**2/epsilon)

        # Construct the kernel and fit to data.
        mykernel = kernel.Kernel(kernel_type='gaussian', metric=metric,
                                 metric_params=metric_params, epsilon=epsilon,
                                 k=len(x_values))
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

    def test_auto_epsilon_selection(self):
        X = np.arange(100).reshape(-1, 1)
        mykernel = kernel.Kernel(kernel_type='gaussian', metric='euclidean',
                                 epsilon='bgh', k=10)
        mykernel.fit(X)
        assert(mykernel.epsilon_fitted == 1.0)
        assert(mykernel.d == 1.0)


class TestBGHEpsilonSelection(object):
    @pytest.mark.parametrize('k', [10, 30, 100])
    def test_1D_uniform_data(self, k):
        X = np.arange(100).reshape(-1, 1)
        neigh = NearestNeighbors(n_neighbors=k)
        sq_dist = neigh.fit(X).kneighbors_graph(X, mode='distance').data**2.
        epsilons = 2**np.arange(-20., 20.)
        eps, d = kernel.choose_optimal_epsilon_BGH(sq_dist, epsilons)
        assert(eps == 1.0)
        assert(d == 1.0)
