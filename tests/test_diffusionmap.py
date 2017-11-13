import numpy as np
import pytest

from pydiffmap import diffusion_map as dm


class TestDiffusionMap(object):
    # These decorators run the test against all possible y, epsilon values.
    #@pytest.mark.parametrize('y_values', y_values_set)
    #@pytest.mark.parametrize('epsilon', epsilons)
    def test_1Dstrip_evals(self):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = 0.25*np.array([1, 4, 9, 16])
        # Setup data and accuracy threshold
        m = 1000
        X = 2*np.pi*np.random.rand(m)
        data = np.array([X]).transpose()
        THRESH = 3.0/np.sqrt(m)
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=100)
        dmap = mydmap.fit_transform(data)
        test_evals = -4./eps*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs(test_evals - real_evals)/real_evals)
        total_error = np.min(errors_eval)
        assert(total_error < THRESH)

    def test_1Dstrip_evecs(self):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        m = 1000
        X = 2*np.pi*np.random.rand(m)
        data = np.array([X]).transpose()
        THRESH = 0.2/np.sqrt(m)
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=100)
        dmap = mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X),mydmap.evecs[:,k])[0,1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)
