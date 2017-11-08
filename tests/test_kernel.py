import numpy as np
import pytest

from pydiffmap import kernel
from scipy.spatial.distance import cdist

x_values = np.vstack((np.linspace(-1, 1, 11), np.arange(11))).T  # set of X vals
y_values_set = [None, x_values, np.arange(22).reshape(11, 2)]  # all sets of Y's
epsilons = [10., 1., 0.1]  # Possible epsilons


class TestKernel(object):
    # These decorators run the test against all possible y, epsilon values.
    @pytest.mark.parametrize('y_values', y_values_set)
    @pytest.mark.parametrize('epsilon', epsilons)
    def test_variable_y_inputs(self, y_values, epsilon):
        # Setup Reference values
        if y_values is None:
            y_values_ref = x_values
        else:
            y_values_ref = y_values
        pw_distance = cdist(y_values_ref, x_values, metric='sqeuclidean')
        true_values = np.exp(-1.*pw_distance/epsilon)

        mykernel = kernel.Kernel(type='gaussian', metric='euclidean',
                                 epsilon=epsilon, k=len(x_values))
        mykernel.fit(x_values)
        K_matrix = mykernel.compute(y_values).toarray()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(K_matrix)
        print(true_values)
        error_values = (K_matrix-true_values).ravel()
#        print(error_values)
        assert(np.linalg.norm(error_values) < 1E-6)
