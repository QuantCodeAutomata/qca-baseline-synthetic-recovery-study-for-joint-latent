"""
Tests for src/em_estimator.py — Joint EM estimator.

Tests verify:
- Numerical utilities (log-pdf, log-sum-exp normalization)
- E-step and M-step correctness
- Convergence properties (monotone log-likelihood)
- Parameter validity after fitting
- Result container fields
"""

from __future__ import annotations

import numpy as np
import pytest

from src.em_estimator import (
    K,
    COV_RIDGE,
    P_EPS,
    EMRunResult,
    EMEstimatorResult,
    _mvn_logpdf_batch,
    _compute_log_responsibilities,
    _normalize_log_responsibilities,
    _observed_data_loglik,
    _m_step,
    _initialize_from_kmeans,
    _run_em,
    fit_joint_em,
)
from src.dgp import simulate


# ---------------------------------------------------------------------------
# Tests for _mvn_logpdf_batch
# ---------------------------------------------------------------------------

class TestMvnLogpdfBatch:
    def test_output_shape(self):
        """log-pdf batch must return (N,) array."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        lp = _mvn_logpdf_batch(X, np.zeros(2), np.eye(2))
        assert lp.shape == (30,)

    def test_standard_normal_at_origin(self):
        """2D standard normal log-pdf at zero must equal -log(2 pi)."""
        X = np.zeros((1, 2))
        lp = _mvn_logpdf_batch(X, np.zeros(2), np.eye(2))
        expected = -np.log(2 * np.pi)
        assert abs(lp[0] - expected) < 1e-10

    def test_finite_outputs(self):
        """All log-pdf values must be finite for standard input."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 2))
        mu = np.array([0.5, -0.5])
        sigma = np.array([[0.4, 0.1], [0.1, 0.4]])
        lp = _mvn_logpdf_batch(X, mu, sigma)
        assert np.all(np.isfinite(lp))

    def test_matches_scipy_logpdf(self):
        """Must match scipy's multivariate_normal.logpdf within tolerance."""
        from scipy.stats import multivariate_normal
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 2))
        mu = np.array([1.0, -0.5])
        sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
        lp_custom = _mvn_logpdf_batch(X, mu, sigma)
        lp_scipy = multivariate_normal.logpdf(X, mean=mu, cov=sigma)
        np.testing.assert_allclose(lp_custom, lp_scipy, atol=1e-9)


# ---------------------------------------------------------------------------
# Tests for E-step utilities
# ---------------------------------------------------------------------------

class TestEStep:
    def setup_method(self):
        """Set up small well-separated data for fast tests."""
        rng = np.random.default_rng(42)
        self.N = 200
        self.X = np.vstack([
            rng.multivariate_normal([2.0, 0.0], [[0.4, 0.1], [0.1, 0.4]], size=50),
            rng.multivariate_normal([2.0, 2.0], [[0.4, 0.1], [0.1, 0.4]], size=50),
            rng.multivariate_normal([-1.0, 0.0], [[0.45, 0.1], [0.1, 0.45]], size=50),
            rng.multivariate_normal([-1.0, 2.0], [[0.45, 0.1], [0.1, 0.45]], size=50),
        ])
        self.D_obs = np.array(
            [0]*48 + [1]*2 + [0]*47 + [1]*3 + [0]*45 + [1]*5 + [0]*37 + [1]*13
        )
        self.pi = np.array([0.25, 0.25, 0.25, 0.25])
        self.mu = np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, 0.0], [-1.0, 2.0]])
        self.sigma = [
            np.array([[0.4, 0.1], [0.1, 0.4]]) for _ in range(2)
        ] + [
            np.array([[0.45, 0.1], [0.1, 0.45]]) for _ in range(2)
        ]
        self.p = np.array([0.02, 0.05, 0.10, 0.25])

    def test_log_responsibilities_shape(self):
        """Log-responsibilities must be (N, K)."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        assert log_tau.shape == (self.N, K)

    def test_log_responsibilities_finite(self):
        """Log-responsibilities must be finite."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        assert np.all(np.isfinite(log_tau))

    def test_normalized_responsibilities_sum_to_one(self):
        """Normalized tau must sum to 1 per row."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        tau, _ = _normalize_log_responsibilities(log_tau)
        np.testing.assert_allclose(tau.sum(axis=1), np.ones(self.N), atol=1e-9)

    def test_normalized_responsibilities_nonnegative(self):
        """Normalized responsibilities must be non-negative."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        tau, _ = _normalize_log_responsibilities(log_tau)
        assert np.all(tau >= 0)

    def test_loglik_is_scalar(self):
        """Observed-data log-likelihood must return a finite scalar."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        ll = _observed_data_loglik(log_tau)
        assert np.isscalar(ll) or ll.ndim == 0
        assert np.isfinite(ll)

    def test_loglik_is_negative(self):
        """Observed-data log-likelihood must be negative (per borrower < 0)."""
        log_tau = _compute_log_responsibilities(
            self.X, self.D_obs, self.pi, self.mu, self.sigma, self.p
        )
        ll = _observed_data_loglik(log_tau)
        assert ll < 0


# ---------------------------------------------------------------------------
# Tests for M-step
# ---------------------------------------------------------------------------

class TestMStep:
    def setup_method(self):
        rng = np.random.default_rng(10)
        self.N = 200
        self.X = rng.standard_normal((self.N, 2))
        self.D_obs = rng.integers(0, 2, size=self.N)
        # Uniform responsibilities (all equal)
        self.tau_uniform = np.ones((self.N, K)) / K

    def test_pi_sums_to_one(self):
        """Estimated pi must sum to 1."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_pi_nonnegative(self):
        """Estimated pi must be non-negative."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        assert np.all(pi >= 0)

    def test_p_in_clipped_range(self):
        """Estimated p must be clipped to [P_EPS, 1-P_EPS]."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        assert np.all(p >= P_EPS)
        assert np.all(p <= 1 - P_EPS)

    def test_sigma_positive_definite(self):
        """Each estimated covariance must be symmetric positive definite."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        for k in range(K):
            np.testing.assert_array_almost_equal(sigma[k], sigma[k].T, decimal=12)
            eigenvals = np.linalg.eigvalsh(sigma[k])
            assert np.all(eigenvals > 0), f"Sigma[{k}] not positive definite"

    def test_mu_shape(self):
        """Estimated mu must be (K, D)."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        assert mu.shape == (K, 2)

    def test_uniform_tau_gives_global_mean(self):
        """With uniform tau, each mu_k must equal the global data mean."""
        pi, mu, sigma, p = _m_step(self.X, self.D_obs, self.tau_uniform)
        global_mean = self.X.mean(axis=0)
        for k in range(K):
            np.testing.assert_allclose(mu[k], global_mean, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for _initialize_from_kmeans
# ---------------------------------------------------------------------------

class TestInitialization:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = np.vstack([
            rng.multivariate_normal([2.0, 0.0], [[0.4, 0.1], [0.1, 0.4]], size=50),
            rng.multivariate_normal([2.0, 2.0], [[0.4, 0.1], [0.1, 0.4]], size=50),
            rng.multivariate_normal([-1.0, 0.0], [[0.45, 0.1], [0.1, 0.45]], size=50),
            rng.multivariate_normal([-1.0, 2.0], [[0.45, 0.1], [0.1, 0.45]], size=50),
        ])
        self.D_obs = np.zeros(200, dtype=int)

    def test_pi_sums_to_one(self):
        pi, mu, sigma, p = _initialize_from_kmeans(self.X, self.D_obs, run_seed=0)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_pi_nonnegative(self):
        pi, mu, sigma, p = _initialize_from_kmeans(self.X, self.D_obs, run_seed=0)
        assert np.all(pi >= 0)

    def test_p_in_valid_range(self):
        pi, mu, sigma, p = _initialize_from_kmeans(self.X, self.D_obs, run_seed=0)
        assert np.all(p >= P_EPS)
        assert np.all(p <= 1 - P_EPS)

    def test_sigma_shapes(self):
        pi, mu, sigma, p = _initialize_from_kmeans(self.X, self.D_obs, run_seed=0)
        assert len(sigma) == K
        for k in range(K):
            assert sigma[k].shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests for _run_em (single run, small data)
# ---------------------------------------------------------------------------

class TestSingleEMRun:
    def setup_method(self):
        """Use small well-separated data for a fast single EM run."""
        rng = np.random.default_rng(42)
        self.X = np.vstack([
            rng.multivariate_normal([2.0, 0.0], [[0.4, 0.1], [0.1, 0.4]], size=300),
            rng.multivariate_normal([2.0, 2.0], [[0.4, 0.1], [0.1, 0.4]], size=300),
            rng.multivariate_normal([-1.0, 0.0], [[0.45, 0.1], [0.1, 0.45]], size=300),
            rng.multivariate_normal([-1.0, 2.0], [[0.45, 0.1], [0.1, 0.45]], size=300),
        ])
        self.D_obs = np.array(
            [0]*295 + [1]*5 + [0]*280 + [1]*20 + [0]*270 + [1]*30 + [0]*225 + [1]*75
        )
        pi_init = np.array([0.25, 0.25, 0.25, 0.25])
        mu_init = np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, 0.0], [-1.0, 2.0]])
        sigma_init = [np.eye(2) * 0.5 for _ in range(4)]
        p_init = np.array([0.02, 0.06, 0.10, 0.25])
        self.result = _run_em(
            self.X, self.D_obs, pi_init, mu_init, sigma_init, p_init
        )

    def test_result_type(self):
        assert isinstance(self.result, EMRunResult)

    def test_pi_sums_to_one(self):
        assert abs(self.result.pi.sum() - 1.0) < 1e-9

    def test_p_in_valid_range(self):
        assert np.all(self.result.p >= P_EPS)
        assert np.all(self.result.p <= 1 - P_EPS)

    def test_tau_shape(self):
        assert self.result.tau.shape == (1200, K)

    def test_tau_rows_sum_to_one(self):
        np.testing.assert_allclose(
            self.result.tau.sum(axis=1), np.ones(1200), atol=1e-9
        )

    def test_log_likelihood_is_finite(self):
        assert np.isfinite(self.result.log_likelihood)

    def test_monotone_log_likelihood(self):
        """EM log-likelihood must be non-decreasing (up to floating-point noise)."""
        ll_hist = self.result.ll_history
        if len(ll_hist) < 2:
            pytest.skip("Not enough iterations to test monotonicity")
        for t in range(1, len(ll_hist)):
            assert ll_hist[t] >= ll_hist[t-1] - 1e-4, \
                f"LL decreased at iteration {t}: {ll_hist[t-1]:.6f} -> {ll_hist[t]:.6f}"

    def test_sigma_positive_definite(self):
        for k in range(K):
            eigenvals = np.linalg.eigvalsh(self.result.sigma[k])
            assert np.all(eigenvals > 0)

    def test_converged(self):
        """Well-separated data should converge."""
        assert self.result.converged


# ---------------------------------------------------------------------------
# Tests for fit_joint_em (integration test with real simulated data, small)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def em_result_module():
    """Module-scoped: run EM once on N=1000 simulated data for integration tests."""
    data = simulate(N=1000, seed=42)
    return fit_joint_em(data.X, data.D, base_seed=0, verbose=False)


class TestFitJointEM:
    def test_result_type(self, em_result_module):
        assert isinstance(em_result_module, EMEstimatorResult)

    def test_at_least_one_admissible_run(self, em_result_module):
        """Must have at least one admissible run."""
        assert em_result_module.n_admissible_runs >= 1

    def test_pi_sums_to_one(self, em_result_module):
        assert abs(em_result_module.pi.sum() - 1.0) < 1e-9

    def test_pi_nonnegative(self, em_result_module):
        assert np.all(em_result_module.pi >= 0)

    def test_p_in_valid_range(self, em_result_module):
        assert np.all(em_result_module.p >= P_EPS)
        assert np.all(em_result_module.p <= 1 - P_EPS)

    def test_tau_rows_sum_to_one(self, em_result_module):
        np.testing.assert_allclose(
            em_result_module.tau.sum(axis=1), np.ones(1000), atol=1e-8
        )

    def test_all_runs_stored(self, em_result_module):
        assert len(em_result_module.all_runs) == 20

    def test_sigma_shapes(self, em_result_module):
        assert len(em_result_module.sigma) == K
        for k in range(K):
            assert em_result_module.sigma[k].shape == (2, 2)
