import numpy as np

from pydiffmap import diffusion_map as dm
from scipy.sparse import csr_matrix

np.random.seed(100)


class TestDiffusionMap(object):
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
        mydmap.fit_transform(data)
        test_evals = -4./eps*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
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
        THRESH = 0.3/np.sqrt(m)
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=100)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_1Dstrip_nonunif_evals(self):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = 0.25*np.array([1, 4, 9, 16])
        # Setup data and accuracy threshold
        m = 1000
        X = np.random.rand(m)
        X = X**2
        X = 2*np.pi*X
        data = np.array([X]).transpose()
        THRESH = 3.0/np.sqrt(m)
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=200)
        mydmap.fit_transform(data)
        test_evals = -4./eps*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    def test_1Dstrip_nonunif_evecs(self):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        m = 1000
        X = np.random.rand(m)
        X = X**2
        X = 2*np.pi*X
        data = np.array([X]).transpose()
        THRESH = 0.3/np.sqrt(m)
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=200)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_2Dstrip_evals(self):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = kx^2 + ky^2 for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        real_evals = 0.25*np.array([1, 4, 4, 5])
        # Setup data and accuracy threshold
        m = 5000
        X = 2.0*np.pi*np.random.rand(m)
        Y = 1.0*np.pi*np.random.rand(m)
        data = np.array([X, Y]).transpose()
        THRESH = 5*3.0/np.sqrt(m)

        eps = 0.02
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=200)
        mydmap.fit_transform(data)
        test_evals = -4./eps*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    def test_2Dstrip_evecs(self):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(kx*x)*cos(ky*y) for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        # Setup data and accuracy threshold
        m = 5000
        X = 2.0*np.pi*np.random.rand(m)
        Y = 1.0*np.pi*np.random.rand(m)
        data = np.array([X, Y]).transpose()
        THRESH = 0.3/np.sqrt(m)

        eps = 0.05
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=200)
        mydmap.fit_transform(data)
        errors_evec = []
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X), mydmap.evecs[:, 0])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(Y), mydmap.evecs[:, 1])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*2*X), mydmap.evecs[:, 2])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X)*np.cos(Y), mydmap.evecs[:, 3])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)


class TestSymmetrization():
    test_mat = csr_matrix([[0, 2.], [0, 3.]])

    def test_and_symmetrization(self):
        ref_mat = np.array([[0, 0], [0, 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='and')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)

    def test_or_symmetrization(self):
        ref_mat = np.array([[0, 2.], [2., 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='or')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)

    def test_avg_symmetrization(self):
        ref_mat = np.array([[0, 1.], [1., 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='average')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)
