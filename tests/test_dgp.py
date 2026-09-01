"""
Tests for src/dgp.py — Data Generating Process module.

Tests verify:
- Simulation shapes and types
- True parameter values match paper specification
- Posterior computation (rows sum to 1, non-negative)
- PD computation in valid range
- Reproducibility with seed
- Edge cases (small N, different seeds)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dgp import (
    TRUE_PI,
    TRUE_MU,
    TRUE_SIGMA,
    TRUE_P,
    PROFILE_DIMS,
    TrueParameters,
    SimulatedData,
    mvn_logpdf,
    compute_true_score_posteriors,
    simulate,
)


# ---------------------------------------------------------------------------
# Tests for true parameter values
# ---------------------------------------------------------------------------

class TestTrueParameters:
    def test_pi_values(self):
        """True mixture weights must match paper exactly."""
        expected = np.array([0.40, 0.20, 0.25, 0.15])
        np.testing.assert_array_almost_equal(TRUE_PI, expected)

    def test_pi_sums_to_one(self):
        """Mixture weights must sum to 1."""
        assert abs(TRUE_PI.sum() - 1.0) < 1e-10

    def test_pi_nonnegative(self):
        """Mixture weights must be non-negative."""
        assert np.all(TRUE_PI >= 0)

    def test_mu_shape(self):
        """Means must be (4, 2) = K profiles × 2 score dimensions."""
        assert TRUE_MU.shape == (4, 2)

    def test_mu_values(self):
        """True means must match paper exactly."""
        expected = np.array([
            [1.8,  0.3],
            [1.8,  1.8],
            [-0.8, 0.3],
            [-0.8, 1.8],
        ])
        np.testing.assert_array_almost_equal(TRUE_MU, expected)

    def test_sigma_positive_definite(self):
        """All covariance matrices must be symmetric positive definite."""
        for k, sig in enumerate(TRUE_SIGMA):
            assert sig.shape == (2, 2), f"Sigma[{k}] shape wrong"
            np.testing.assert_array_almost_equal(sig, sig.T, err_msg=f"Sigma[{k}] not symmetric")
            eigenvals = np.linalg.eigvalsh(sig)
            assert np.all(eigenvals > 0), f"Sigma[{k}] not positive definite"

    def test_sigma_values_profiles_1_2(self):
        """Profiles 1 and 2 must share the same covariance."""
        expected = np.array([[0.40, 0.10], [0.10, 0.40]])
        np.testing.assert_array_almost_equal(TRUE_SIGMA[0], expected)
        np.testing.assert_array_almost_equal(TRUE_SIGMA[1], expected)

    def test_sigma_values_profiles_3_4(self):
        """Profiles 3 and 4 must share the same covariance."""
        expected = np.array([[0.45, 0.10], [0.10, 0.45]])
        np.testing.assert_array_almost_equal(TRUE_SIGMA[2], expected)
        np.testing.assert_array_almost_equal(TRUE_SIGMA[3], expected)

    def test_p_values(self):
        """True default probabilities must match paper exactly."""
        expected = np.array([0.01, 0.05, 0.10, 0.25])
        np.testing.assert_array_almost_equal(TRUE_P, expected)

    def test_p_in_unit_interval(self):
        """Default probabilities must be in (0, 1)."""
        assert np.all(TRUE_P > 0)
        assert np.all(TRUE_P < 1)

    def test_profile_dims(self):
        """Profile-to-latent-dimension mapping must follow paper ordering."""
        expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(PROFILE_DIMS, expected)

    def test_true_parameters_dataclass(self):
        """TrueParameters dataclass must have correct attributes."""
        theta = TrueParameters()
        assert theta.K == 4
        assert theta.D == 2
        np.testing.assert_array_equal(theta.pi, TRUE_PI)


# ---------------------------------------------------------------------------
# Tests for mvn_logpdf
# ---------------------------------------------------------------------------

class TestMvnLogpdf:
    def test_output_shape(self):
        """log-pdf must return (N,) array."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        mu = np.zeros(2)
        sigma = np.eye(2)
        lp = mvn_logpdf(X, mu, sigma)
        assert lp.shape == (50,)

    def test_standard_normal_at_zero(self):
        """Standard 2D Gaussian log-pdf at zero must equal -log(2*pi)."""
        X = np.zeros((1, 2))
        lp = mvn_logpdf(X, np.zeros(2), np.eye(2))
        expected = -np.log(2 * np.pi)
        assert abs(lp[0] - expected) < 1e-10

    def test_returns_finite_values(self):
        """log-pdf must return finite values for valid inputs."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        lp = mvn_logpdf(X, np.array([0.5, -0.5]), np.array([[0.4, 0.1], [0.1, 0.4]]))
        assert np.all(np.isfinite(lp))


# ---------------------------------------------------------------------------
# Tests for compute_true_score_posteriors
# ---------------------------------------------------------------------------

class TestTrueScorePosteriors:
    def setup_method(self):
        self.theta = TrueParameters()
        rng = np.random.default_rng(99)
        self.X = rng.multivariate_normal([1.8, 0.3], [[0.4, 0.1], [0.1, 0.4]], size=100)

    def test_output_shape(self):
        """Posteriors must be (N, K) = (100, 4)."""
        gamma = compute_true_score_posteriors(self.X, self.theta)
        assert gamma.shape == (100, 4)

    def test_rows_sum_to_one(self):
        """Each posterior row must sum to 1 (within floating-point tolerance)."""
        gamma = compute_true_score_posteriors(self.X, self.theta)
        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(100), atol=1e-9)

    def test_nonnegative(self):
        """All posterior values must be non-negative."""
        gamma = compute_true_score_posteriors(self.X, self.theta)
        assert np.all(gamma >= 0)

    def test_bounded_above_by_one(self):
        """No posterior value may exceed 1."""
        gamma = compute_true_score_posteriors(self.X, self.theta)
        assert np.all(gamma <= 1 + 1e-10)


# ---------------------------------------------------------------------------
# Tests for simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def setup_method(self):
        self.data = simulate(N=5000, seed=1234)

    def test_output_type(self):
        """simulate must return SimulatedData."""
        assert isinstance(self.data, SimulatedData)

    def test_X_shape(self):
        """Score matrix must be (5000, 2)."""
        assert self.data.X.shape == (5000, 2)

    def test_D_shape(self):
        """Default indicators must have shape (5000,)."""
        assert self.data.D.shape == (5000,)

    def test_D_binary(self):
        """Default indicators must be binary (0 or 1)."""
        assert set(np.unique(self.data.D)).issubset({0, 1})

    def test_Z_values(self):
        """Latent profile indices must be in {0, 1, 2, 3}."""
        assert set(np.unique(self.data.Z)).issubset({0, 1, 2, 3})

    def test_Z_R_binary(self):
        """Z_R must be binary (0 or 1)."""
        assert set(np.unique(self.data.Z_R)).issubset({0, 1})

    def test_Z_F_binary(self):
        """Z_F must be binary (0 or 1)."""
        assert set(np.unique(self.data.Z_F)).issubset({0, 1})

    def test_gamma_X_true_shape(self):
        """True score posteriors must be (5000, 4)."""
        assert self.data.gamma_X_true.shape == (5000, 4)

    def test_gamma_X_true_rows_sum_to_one(self):
        """True posterior rows must sum to 1."""
        np.testing.assert_allclose(
            self.data.gamma_X_true.sum(axis=1), np.ones(5000), atol=1e-9
        )

    def test_PD_true_shape(self):
        """True PDs must have shape (5000,)."""
        assert self.data.PD_true.shape == (5000,)

    def test_PD_true_in_unit_interval(self):
        """True PDs must be in [0, 1]."""
        assert np.all(self.data.PD_true >= 0)
        assert np.all(self.data.PD_true <= 1)

    def test_PD_true_range_matches_true_p(self):
        """True PDs must be bounded by min and max true p_k values."""
        theta = TrueParameters()
        assert self.data.PD_true.min() >= theta.p.min() - 1e-10
        assert self.data.PD_true.max() <= theta.p.max() + 1e-10

    def test_reproducibility(self):
        """simulate with same seed must produce identical results."""
        data2 = simulate(N=5000, seed=1234)
        np.testing.assert_array_equal(self.data.X, data2.X)
        np.testing.assert_array_equal(self.data.D, data2.D)
        np.testing.assert_array_equal(self.data.Z, data2.Z)

    def test_different_seed_gives_different_data(self):
        """Different seeds must produce different data."""
        data2 = simulate(N=5000, seed=9999)
        assert not np.allclose(self.data.X, data2.X)

    def test_small_N(self):
        """simulate must work for small N (edge case)."""
        data_small = simulate(N=50, seed=0)
        assert data_small.X.shape == (50, 2)
        assert data_small.D.shape == (50,)

    def test_profile_dim_consistency(self):
        """Z_R and Z_F must be consistent with Z via PROFILE_DIMS."""
        for i in range(min(200, self.data.N)):
            k = self.data.Z[i]
            assert self.data.Z_R[i] == PROFILE_DIMS[k, 0]
            assert self.data.Z_F[i] == PROFILE_DIMS[k, 1]

    def test_empirical_default_rate_ordering(self):
        """Profiles with higher p_k should have higher empirical default rates."""
        empirical_rates = [self.data.D[self.data.Z == k].mean() for k in range(4)]
        # Profile 1 (p=0.01) should have lowest rate, Profile 4 (p=0.25) highest
        assert empirical_rates[0] < empirical_rates[3]

    def test_approximate_pi_match(self):
        """Empirical profile frequencies should be close to true pi."""
        N = self.data.N
        for k in range(4):
            emp_pi = (self.data.Z == k).sum() / N
            assert abs(emp_pi - TRUE_PI[k]) < 0.05, \
                f"Profile {k}: empirical pi={emp_pi:.3f}, true={TRUE_PI[k]}"
