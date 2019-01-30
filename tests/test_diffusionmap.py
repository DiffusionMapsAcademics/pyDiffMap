import numpy as np
import pytest

from pydiffmap import diffusion_map as dm


class TestDiffusionMap(object):
    @pytest.mark.parametrize('epsilon', [0.002, 'bgh'])
    def test_1Dstrip_evals(self, epsilon):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = -0.25*np.array([1, 4, 9, 16])
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.05
        # Setup diffusion map
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, epsilon=epsilon, alpha=1.0, k=20)
        mydmap.fit(data)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((mydmap.evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)

        assert(total_error < THRESH)

    @pytest.mark.parametrize('epsilon', [0.002, 'bgh'])
    def test_1Dstrip_evecs(self, epsilon):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.003
        # Setup diffusion map
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, epsilon=epsilon, alpha=1.0, k=40)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('epsilon', [0.005, 'bgh'])
    def test_1Dstrip_nonunif_evals(self, epsilon):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = -0.25*np.array([1, 4, 9, 16])
        # Setup data and accuracy threshold
        X = (np.linspace(0., 1., 81)**2)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.1
        # Setup diffusion map
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, epsilon=epsilon, alpha=1.0, k=40)
        mydmap.fit_transform(data)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((mydmap.evals- real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('epsilon', [0.005, 'bgh'])
    def test_1Dstrip_nonunif_evecs(self, epsilon):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        X = (np.linspace(0., 1., 81)**2)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.01
        # Setup diffusion map
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, epsilon=epsilon, alpha=1.0, k=40)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_2Dstrip_evals(self, uniform_2d_data):
        """
        Test that we compute the correct eigenvalues on a 2d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = kx^2 + ky^2 for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        real_evals = -0.25*np.array([1, 4, 4, 5])
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.2

        eps = 0.0025
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, alpha=1.0, k=100, epsilon=eps)
        mydmap.fit(data)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((mydmap.evals- real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    def test_2Dstrip_evecs(self, uniform_2d_data):
        """
        Test that we compute the correct eigenvectors (cosines) on a 2d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(kx*x)*cos(ky*y) for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.01

        eps = 0.0025
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, alpha=1.0, k=100, epsilon=eps)
        mydmap.fit(data)
        errors_evec = []
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X), mydmap.evecs[:, 0])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(Y), mydmap.evecs[:, 1])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*2*X), mydmap.evecs[:, 2])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X)*np.cos(Y), mydmap.evecs[:, 3])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_sphere_evals(self, spherical_data):
        """
        Test that we compute the correct eigenvalues on a 2d sphere embedded in 3d.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        data, Phi, Theta = spherical_data
        # Setup true values to test against.
        real_evals = -1 * np.array([2, 2, 2, 6])  # =l(l+1)
        THRESH = 0.1
        eps = 0.015
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, alpha=1.0, k=400, epsilon=eps)
        mydmap.fit(data)

        # Check eigenvalues pass below error tolerance.
        errors_eval = abs((mydmap.evals- real_evals)/real_evals)
        max_eval_error = np.max(errors_eval)
        assert(max_eval_error < THRESH)

    def test_sphere_evecs(self, spherical_data):
        """
        Test that we compute the correct eigenvectors (spherical harmonics) on a 2d sphere embedded in R^3.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        data, Phi, Theta = spherical_data
        THRESH = 0.001
        eps = 0.015
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=4, alpha=1.0, k=400, epsilon=eps)
        mydmap.fit(data)
        # rotate sphere so that maximum of first DC is at the north pole
        northpole = np.argmax(mydmap.dmap[:, 0])
        phi_n = Phi[northpole]
        theta_n = Theta[northpole]
        R = np.array([[np.sin(theta_n)*np.cos(phi_n), np.sin(theta_n)*np.sin(phi_n), -np.cos(theta_n)],
                      [-np.sin(phi_n), np.cos(phi_n), 0],
                      [np.cos(theta_n)*np.cos(phi_n), np.cos(theta_n)*np.sin(phi_n), np.sin(theta_n)]])
        data_rotated = np.dot(R, data.transpose())
        # check that error is beneath tolerance.
        evec_error = 1 - np.corrcoef(mydmap.dmap[:, 0], data_rotated[2, :])[0, 1]
        assert(evec_error < THRESH)

    def test_explicit_density(self, harmonic_1d_data):
        """
        Test explicit density function.
        This test tests the implementation and is independent on all the other parameters.
        """

        data = harmonic_1d_data
        density_fxn = lambda x: (1.0/(np.sqrt(np.pi * 2))) * np.exp(-0.5 * x**2).squeeze()

        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, epsilon=0.1, alpha=0.5, k=100, density_fxn=density_fxn)
        mydmap.fit(data)

        err = np.max((np.abs(mydmap.q / np.linalg.norm(mydmap.q) - density_fxn(data) / np.linalg.norm(density_fxn(data)))))

        assert(err == 0)

    @pytest.mark.parametrize('epsilon', [0.1, 'bgh'])
    def test_explicit_density_kde(self, harmonic_1d_data, epsilon):
        """
        Test the implicit kernel density estimator. Results depend on knearest neighbors
        and epsilon. This test is not very stable, tolerancy threshold is therefore chosen high.
        """
        THRESH = 0.2
        data = harmonic_1d_data
        # reject_outliers to stabilise
        m = 2
        data = data[abs(data - np.mean(data)) < m * np.std(data), np.newaxis]

        density_fxn = lambda x: (1.0/(np.sqrt(np.pi * 2))) * np.exp(-0.5 * x**2).squeeze()

        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, epsilon=epsilon, alpha=0.5, k=100)
        dmap = mydmap.fit(data)

        true = density_fxn(data) / np.linalg.norm(density_fxn(data))
        kde = mydmap.q / np.linalg.norm(mydmap.q)

        err = np.linalg.norm(true - kde) / np.linalg.norm(kde)

        assert(err < THRESH)

class TestNystroem(object):
    @pytest.mark.parametrize('method', ['nystroem', 'power'])
    def test_2Dstrip_nystroem(self, uniform_2d_data, method):
        """
        Test the nystroem extension in the transform() function.
        """
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.01
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=1, alpha=1.0, k=100, epsilon=eps, oos=method)
        mydmap.fit(data)
        # Setup values to test against (regular grid)
        x_test, y_test = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(0, np.pi, 40))
        X_test = np.array([x_test.ravel(), y_test.ravel()]).transpose()
        # call nystroem extension
        dmap_ext = mydmap.transform(X_test)
        # extract first diffusion coordinate and normalize
        V_test = dmap_ext[:, 0]
        V_test = V_test/np.linalg.norm(V_test)
        # true dominant eigenfunction = cos(0.5*x), normalize
        V_true = np.cos(.5*x_test).ravel()
        V_true = V_true/np.linalg.norm(V_true)
        # compute L2 error, deal with remaining sign ambiguity
        error = min([np.linalg.norm(V_true+V_test), np.linalg.norm(V_true-V_test)])
        assert(error < THRESH)


class TestWeighting(object):
    @pytest.mark.parametrize('epsilon', [0.002, 'bgh'])
    @pytest.mark.parametrize('oos', ['power', 'nystroem', False])
    @pytest.mark.parametrize('dmap_method', ['base', 'TMDmap'])
    def test_1Dstrip_evecs(self, epsilon, oos, dmap_method):
        """
        Test measure reweighting.  We reweight the uniform distribution to
        approximate a Gaussian distribution.  For numerical reasons, we truncate
        the domain to the interval [-5, 5].

        Here, we test eigenvector accuracy.  Eigenvectors should be the
        probabalists Hermite polynomials.
        """
        # Setup data and accuracy threshold
        # X = np.linspace(-5., 5., 201)
        X = np.linspace(0, 2.5, 101)**2
        X = np.hstack([-1 * np.copy(X[1:][::-1]), X])
        if not oos:
            Y = X
            oos = 'nystroem'
        else:
            Y = np.linspace(-5., 5., 101)
        data_x = np.array([X]).transpose()
        data_y = np.array([Y]).transpose()
        EVEC_THRESH = 0.005
        EVAL_THRESH = 0.003
        # Setup true values to test against.
        real_evecs = [Y, Y**2-1, Y**3-3*Y,
                      Y**4-6*Y**2+3]  # Hermite polynomials
        real_evals = -1 * np.arange(1, 5)
        # Setup diffusion map
        if dmap_method == 'TMDmap':
            com_fxn = lambda y_j: np.exp(-.5*np.dot(y_j, y_j))
            mydmap = dm.TMDmap(alpha=1., n_evecs=4, epsilon=epsilon, k=100, change_of_measure=com_fxn, oos=oos)
        else:
            weight_fxn = lambda y_j: np.exp(-.25*np.dot(y_j, y_j))
            mydmap = dm.DiffusionMap.from_sklearn(alpha=1., n_evecs=4, epsilon=epsilon, k=100, weight_fxn=weight_fxn, oos=oos)

        # Fit data and build dmap
        mydmap.fit(data_x)
        evecs = mydmap.transform(data_y)
        errors_evec = []
        for k in range(4):
            errors_evec.append(abs(np.corrcoef(real_evecs[k], evecs[:, k])[0, 1]))

        # Check that relative evec error values are beneath tolerance.
        total_evec_error = 1 - np.min(errors_evec)
        assert(total_evec_error < EVEC_THRESH)
        # Check that relative eval error values are beneath tolerance.
        errors_eval = abs((mydmap.evals- real_evals)/real_evals)
        total_eval_error = np.min(errors_eval)
        assert(total_eval_error < EVAL_THRESH)


class TestBandwidths(object):
    @pytest.mark.parametrize('alpha_beta', [(0., -1./3), (-1./4, -1./2)])
    @pytest.mark.parametrize('explicit_bandwidth', [False, True])
    def test_bandwidth_norm(self, harmonic_1d_data, alpha_beta, explicit_bandwidth):
        data = harmonic_1d_data
        alpha, beta = alpha_beta
        X = data[:, 0]
        THRESHS = np.array([0.01, 0.01, 0.1])
        ref_evecs = [X, X**2, (X**3 - 3 * X)/np.sqrt(6)]

        if explicit_bandwidth:
            bandwidth_type = lambda x: np.exp(-1. * x[:, 0]**2 * (beta / 2.))  # bandwidth is density^beta
        else:
            bandwidth_type = beta

        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=3, epsilon='bgh', alpha=alpha,
                                 k=50, bandwidth_type=bandwidth_type, bandwidth_normalize=True)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(3):
            errors_evec.append(abs(np.corrcoef(ref_evecs[k], mydmap.evecs[:, k])[0, 1]))
        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.array(errors_evec)
        assert((total_error < THRESHS).all())

    @pytest.mark.parametrize('alpha_beta', [(0., -1./3), (-1./4, -1./2)])
    @pytest.mark.parametrize('explicit_bandwidth', [False, True])
    def test_bandwidth_norm_oos(self, harmonic_1d_data, alpha_beta, explicit_bandwidth):
        data = harmonic_1d_data
        alpha, beta = alpha_beta
        oos_data = np.linspace(-1.5, 1.5, 51).reshape(-1, 1)
        Y = oos_data.ravel()
        THRESHS = np.array([0.01, 0.01, 0.1])
        ref_evecs = [Y, Y**2, (Y**3 - 3 * Y)/np.sqrt(6)]

        if explicit_bandwidth:
            bandwidth_type = lambda x: np.exp(-1. * x[:, 0]**2 * (beta / 2.))  # bandwidth is density^beta
        else:
            bandwidth_type = beta
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=3, epsilon='bgh', alpha=alpha,
                                 k=50, bandwidth_type=bandwidth_type, bandwidth_normalize=True,
                                 oos='power')
        mydmap.fit(data)
        oos_evecs = mydmap.transform(oos_data)
        errors_evec = []
        for k in np.arange(3):
            errors_evec.append(abs(np.corrcoef(ref_evecs[k], oos_evecs[:, k])[0, 1]))
        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.array(errors_evec)
        assert((total_error < THRESHS).all())
