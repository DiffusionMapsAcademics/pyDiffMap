import numpy as np
import pytest

from pydiffmap import kernel
from scipy.spatial.distance import cdist

x_values = np.vstack((np.linspace(-1, 1, 11), np.arange(11))).T  # set of X vals
y_values_set = [None, x_values, np.arange(6).reshape(-1,2), np.arange(44).reshape(-1, 2)]  # all sets of Y's
epsilons = [10., 1., 0.1]  # Possible epsilons


class TestKernel(object):
    # These decorators run the test against all possible y, epsilon values.
    @pytest.mark.parametrize('y_values', y_values_set)
    @pytest.mark.parametrize('epsilon', epsilons)
    def test_matrix_output(self, y_values, epsilon):
        """
        Test that we are returning the correct kernel values.
        """
        # Setup true values to test again.
        if y_values is None:
            y_values_ref = x_values
        else:
            y_values_ref = y_values
        pw_distance = cdist(y_values_ref, x_values, metric='sqeuclidean')
        true_values = np.exp(-1.*pw_distance/epsilon)

        # Construct the kernel and fit to data.
        mykernel = kernel.Kernel(type='gaussian', metric='euclidean',
                                 epsilon=epsilon, k=len(x_values))
        mykernel.fit(x_values)
        K_matrix = mykernel.compute(y_values).toarray()

        # Check that error values are beneath tolerance.
        error_values = (K_matrix-true_values).ravel()
        total_error = np.linalg.norm(error_values)
        assert(total_error < 1E-8)

    @pytest.mark.parametrize('k', np.arange(2,14,2))
    def test_neighborlists(self,k):
        """
        Test that neighborlisting gives the right number of elements.
        """
        # Correct number of nearest neighbors.
        k0 = min(k,len(x_values))
        
        # Construct kernel matrix.
        mykernel = kernel.Kernel(type='gaussian', metric='euclidean',
                                 epsilon=1., k=k)
        mykernel.fit(x_values)
        K_matrix = mykernel.compute(x_values)

        # Check if each row has correct number of elements
        row_has_k_elements = (K_matrix.nnz == k0*len(x_values))
        assert(row_has_k_elements)
