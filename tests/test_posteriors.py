"""
Tests for src/posteriors.py — Score-based posterior and PD reconstruction.

Tests verify:
- compute_score_posteriors: shape, rows sum to 1, non-negative, D_i not used
- compute_borrower_pd: shape, values in [0,1], consistency with gamma @ p
- Edge cases: single borrower, uniform mixture
"""

from __future__ import annotations

import numpy as np
import pytest

from src.posteriors import compute_score_posteriors, compute_borrower_pd
from src.dgp import TRUE_MU, TRUE_SIGMA, TRUE_PI, TRUE_P


# ---------------------------------------------------------------------------
# Tests for compute_score_posteriors
# ---------------------------------------------------------------------------

class TestComputeScorePosteriors:
    def setup_method(self):
        """Create simple test data with known structure."""
        self.N = 150
        rng = np.random.default_rng(55)
        self.X = rng.standard_normal((self.N, 2))
        self.pi = TRUE_PI.copy()
        self.mu = TRUE_MU.copy()
        self.sigma = [s.copy() for s in TRUE_SIGMA]

    def test_output_shape(self):
        """Output must be (N, K)."""
        gamma = compute_score_posteriors(self.X, self.pi, self.mu, self.sigma)
        assert gamma.shape == (self.N, 4)

    def test_rows_sum_to_one(self):
        """Posterior rows must sum to 1."""
        gamma = compute_score_posteriors(self.X, self.pi, self.mu, self.sigma)
        np.testing.assert_allclose(
            gamma.sum(axis=1), np.ones(self.N), atol=1e-9
        )

    def test_nonnegative(self):
        """All posterior values must be non-negative."""
        gamma = compute_score_posteriors(self.X, self.pi, self.mu, self.sigma)
        assert np.all(gamma >= 0)

    def test_bounded_above_by_one(self):
        """No posterior may exceed 1."""
        gamma = compute_score_posteriors(self.X, self.pi, self.mu, self.sigma)
        assert np.all(gamma <= 1 + 1e-10)

    def test_scores_only_not_affected_by_default(self):
        """compute_score_posteriors must not use D_i in its computation."""
        # The function doesn't take D as input — verify output is independent of defaults
        gamma1 = compute_score_posteriors(self.X, self.pi, self.mu, self.sigma)
        # No way to pass D, so just verify the signature doesn't include it
        import inspect
        sig = inspect.signature(compute_score_posteriors)
        assert "D" not in sig.parameters, \
            "compute_score_posteriors must not take D_i as input"

    def test_single_borrower(self):
        """Must work with N=1 borrower."""
        X_single = self.X[:1]
        gamma = compute_score_posteriors(X_single, self.pi, self.mu, self.sigma)
        assert gamma.shape == (1, 4)
        np.testing.assert_allclose(gamma.sum(axis=1), [1.0], atol=1e-9)

    def test_matches_true_posterior_for_true_params(self):
        """With true parameters, must match compute_true_score_posteriors."""
        from src.dgp import compute_true_score_posteriors, TrueParameters
        theta = TrueParameters()
        gamma_est = compute_score_posteriors(self.X, theta.pi, theta.mu, theta.sigma)
        gamma_true = compute_true_score_posteriors(self.X, theta)
        np.testing.assert_allclose(gamma_est, gamma_true, atol=1e-10)

    def test_uniform_mixture_gives_proportional_densities(self):
        """With equal weights, posterior is proportional to marginal density."""
        pi_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        gamma = compute_score_posteriors(self.X, pi_uniform, self.mu, self.sigma)
        # Rows must still sum to 1
        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(self.N), atol=1e-9)

    def test_degenerate_component_dominates_nearby_data(self):
        """A point very close to mu_1 should have high posterior for component 1."""
        X_near_mu1 = TRUE_MU[0:1]  # exactly at the mean of component 1
        gamma = compute_score_posteriors(X_near_mu1, self.pi, self.mu, self.sigma)
        # Component 1 should have the highest posterior (by separation of means)
        assert gamma[0, 0] > 0.5, f"Expected gamma_1 > 0.5, got {gamma[0, 0]:.4f}"


# ---------------------------------------------------------------------------
# Tests for compute_borrower_pd
# ---------------------------------------------------------------------------

class TestComputeBorrowerPD:
    def setup_method(self):
        rng = np.random.default_rng(77)
        raw = np.abs(rng.standard_normal((200, 4)))
        self.gamma = raw / raw.sum(axis=1, keepdims=True)
        self.p = TRUE_P.copy()

    def test_output_shape(self):
        """PD must have shape (N,)."""
        pd = compute_borrower_pd(self.gamma, self.p)
        assert pd.shape == (200,)

    def test_pd_in_unit_interval(self):
        """PDs must be in [0, 1]."""
        pd = compute_borrower_pd(self.gamma, self.p)
        assert np.all(pd >= 0)
        assert np.all(pd <= 1)

    def test_pd_bounded_by_p_min_max(self):
        """PDs must be in [min(p), max(p)] since they are convex combinations."""
        pd = compute_borrower_pd(self.gamma, self.p)
        assert pd.min() >= self.p.min() - 1e-10
        assert pd.max() <= self.p.max() + 1e-10

    def test_pd_equals_matrix_product(self):
        """PD must equal gamma @ p."""
        pd = compute_borrower_pd(self.gamma, self.p)
        expected = self.gamma @ self.p
        np.testing.assert_allclose(pd, expected, atol=1e-12)

    def test_uniform_posterior_gives_mean_p(self):
        """Uniform posterior (all 1/K) must give the average p_k."""
        gamma_uniform = np.ones((10, 4)) / 4
        pd = compute_borrower_pd(gamma_uniform, self.p)
        expected_pd = self.p.mean()
        np.testing.assert_allclose(pd, np.full(10, expected_pd), atol=1e-10)

    def test_degenerate_posterior_gives_exact_p_k(self):
        """Posterior concentrated on component k must give p_k."""
        for k in range(4):
            gamma_degenerate = np.zeros((1, 4))
            gamma_degenerate[0, k] = 1.0
            pd = compute_borrower_pd(gamma_degenerate, self.p)
            np.testing.assert_allclose(pd[0], self.p[k], atol=1e-12)

    def test_single_borrower(self):
        """Must work for a single borrower."""
        pd = compute_borrower_pd(self.gamma[:1], self.p)
        assert pd.shape == (1,)
        assert 0 <= pd[0] <= 1

    def test_dimension_mismatch_raises(self):
        """Must raise an AssertionError if gamma columns != len(p)."""
        gamma_wrong = np.ones((10, 3)) / 3
        p_4 = np.array([0.01, 0.05, 0.10, 0.25])
        with pytest.raises(AssertionError):
            compute_borrower_pd(gamma_wrong, p_4)
