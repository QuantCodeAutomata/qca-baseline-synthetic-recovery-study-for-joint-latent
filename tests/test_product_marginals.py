"""
Tests for src/product_marginals.py — Product-of-marginals analysis.

Tests verify:
- compute_posterior_marginals: marginal shapes, values, sums
- compute_product_marginals: rows sum to 1, non-negative, independence property
- kl_joint_vs_product: non-negative, zero for independent joints, 0*log(0/q)=0
- pd_from_joint / pd_from_product: range, consistency
- summarize_pm_metrics: correct metric types and bounds
"""

from __future__ import annotations

import numpy as np
import pytest

from src.product_marginals import (
    compute_posterior_marginals,
    compute_product_marginals,
    kl_joint_vs_product,
    pd_from_joint,
    pd_from_product,
    summarize_pm_metrics,
    ProductMarginalsMetrics,
)
from src.dgp import TRUE_P


# ---------------------------------------------------------------------------
# Helper: make valid gamma arrays
# ---------------------------------------------------------------------------

def _make_gamma(N: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.standard_normal((N, 4))) + 0.01
    return raw / raw.sum(axis=1, keepdims=True)


def _make_independent_gamma(N: int = 100, seed: int = 42) -> np.ndarray:
    """Create a gamma that IS the product of its marginals (truly independent)."""
    rng = np.random.default_rng(seed)
    p_R = rng.dirichlet([1, 1], size=N)   # (N, 2)
    p_F = rng.dirichlet([1, 1], size=N)   # (N, 2)
    # gamma = product of marginals
    gamma = np.column_stack([
        p_R[:, 0] * p_F[:, 0],
        p_R[:, 0] * p_F[:, 1],
        p_R[:, 1] * p_F[:, 0],
        p_R[:, 1] * p_F[:, 1],
    ])
    return gamma


# ---------------------------------------------------------------------------
# Tests for compute_posterior_marginals
# ---------------------------------------------------------------------------

class TestComputePosteriorMarginals:
    def setup_method(self):
        self.gamma = _make_gamma(100)

    def test_marg_R_shape(self):
        marg_R, marg_F = compute_posterior_marginals(self.gamma)
        assert marg_R.shape == (100, 2)

    def test_marg_F_shape(self):
        marg_R, marg_F = compute_posterior_marginals(self.gamma)
        assert marg_F.shape == (100, 2)

    def test_marg_R_rows_sum_to_one(self):
        marg_R, marg_F = compute_posterior_marginals(self.gamma)
        np.testing.assert_allclose(marg_R.sum(axis=1), np.ones(100), atol=1e-10)

    def test_marg_F_rows_sum_to_one(self):
        marg_R, marg_F = compute_posterior_marginals(self.gamma)
        np.testing.assert_allclose(marg_F.sum(axis=1), np.ones(100), atol=1e-10)

    def test_marg_R_nonnegative(self):
        marg_R, _ = compute_posterior_marginals(self.gamma)
        assert np.all(marg_R >= 0)

    def test_marg_F_nonnegative(self):
        _, marg_F = compute_posterior_marginals(self.gamma)
        assert np.all(marg_F >= 0)

    def test_marg_R_correct_formulas(self):
        """marg_R[:,0] = gamma[:,0]+gamma[:,1], marg_R[:,1] = gamma[:,2]+gamma[:,3]."""
        marg_R, _ = compute_posterior_marginals(self.gamma)
        np.testing.assert_allclose(
            marg_R[:, 0], self.gamma[:, 0] + self.gamma[:, 1], atol=1e-12
        )
        np.testing.assert_allclose(
            marg_R[:, 1], self.gamma[:, 2] + self.gamma[:, 3], atol=1e-12
        )

    def test_marg_F_correct_formulas(self):
        """marg_F[:,0] = gamma[:,0]+gamma[:,2], marg_F[:,1] = gamma[:,1]+gamma[:,3]."""
        _, marg_F = compute_posterior_marginals(self.gamma)
        np.testing.assert_allclose(
            marg_F[:, 0], self.gamma[:, 0] + self.gamma[:, 2], atol=1e-12
        )
        np.testing.assert_allclose(
            marg_F[:, 1], self.gamma[:, 1] + self.gamma[:, 3], atol=1e-12
        )

    def test_wrong_shape_raises(self):
        """Must raise AssertionError with wrong number of columns."""
        gamma_wrong = np.ones((10, 3)) / 3
        with pytest.raises(AssertionError):
            compute_posterior_marginals(gamma_wrong)


# ---------------------------------------------------------------------------
# Tests for compute_product_marginals
# ---------------------------------------------------------------------------

class TestComputeProductMarginals:
    def setup_method(self):
        self.gamma = _make_gamma(100, seed=5)

    def test_output_shape(self):
        q_pm = compute_product_marginals(self.gamma)
        assert q_pm.shape == (100, 4)

    def test_rows_sum_to_one(self):
        q_pm = compute_product_marginals(self.gamma)
        np.testing.assert_allclose(q_pm.sum(axis=1), np.ones(100), atol=1e-10)

    def test_nonnegative(self):
        q_pm = compute_product_marginals(self.gamma)
        assert np.all(q_pm >= 0)

    def test_correct_product_formula(self):
        """q_pm[i,0] must equal marg_R[i,0]*marg_F[i,0]."""
        q_pm = compute_product_marginals(self.gamma)
        marg_R, marg_F = compute_posterior_marginals(self.gamma)
        np.testing.assert_allclose(
            q_pm[:, 0], marg_R[:, 0] * marg_F[:, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            q_pm[:, 3], marg_R[:, 1] * marg_F[:, 1], atol=1e-12
        )

    def test_equals_input_when_already_independent(self):
        """When gamma = product of marginals, q_pm must equal gamma."""
        gamma_indep = _make_independent_gamma(50, seed=7)
        q_pm = compute_product_marginals(gamma_indep)
        np.testing.assert_allclose(q_pm, gamma_indep, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for kl_joint_vs_product
# ---------------------------------------------------------------------------

class TestKlJointVsProduct:
    def test_output_shape(self):
        gamma = _make_gamma(80)
        q_pm = compute_product_marginals(gamma)
        kl = kl_joint_vs_product(gamma, q_pm)
        assert kl.shape == (80,)

    def test_nonnegative(self):
        gamma = _make_gamma(80, seed=10)
        q_pm = compute_product_marginals(gamma)
        kl = kl_joint_vs_product(gamma, q_pm)
        assert np.all(kl >= 0)

    def test_zero_when_independent(self):
        """KL must be (near) zero when gamma is already factorized."""
        gamma_indep = _make_independent_gamma(100, seed=3)
        q_pm = compute_product_marginals(gamma_indep)
        kl = kl_joint_vs_product(gamma_indep, q_pm)
        np.testing.assert_allclose(kl, np.zeros(100), atol=1e-9)

    def test_handles_zero_entries(self):
        """Must not produce NaN when gamma has zero entries (0*log(0/q)=0)."""
        gamma_degenerate = np.array([[1.0, 0.0, 0.0, 0.0]])
        q_pm = compute_product_marginals(gamma_degenerate)
        kl = kl_joint_vs_product(gamma_degenerate, q_pm)
        assert np.all(np.isfinite(kl))
        assert kl[0] >= 0

    def test_positive_for_correlated_posterior(self):
        """For a correlated (non-independent) posterior, KL must be > 0."""
        # Heavily concentrated on (0,0) and (1,1) — positive correlation
        gamma = np.array([[0.48, 0.02, 0.02, 0.48]])
        q_pm = compute_product_marginals(gamma)
        kl = kl_joint_vs_product(gamma, q_pm)
        assert kl[0] > 0, f"Expected positive KL for correlated posterior, got {kl[0]}"


# ---------------------------------------------------------------------------
# Tests for pd_from_joint / pd_from_product
# ---------------------------------------------------------------------------

class TestPdFromJoint:
    def test_output_shape(self):
        gamma = _make_gamma(60)
        pd = pd_from_joint(gamma, TRUE_P)
        assert pd.shape == (60,)

    def test_bounded(self):
        gamma = _make_gamma(60)
        pd = pd_from_joint(gamma, TRUE_P)
        assert np.all(pd >= TRUE_P.min() - 1e-10)
        assert np.all(pd <= TRUE_P.max() + 1e-10)

    def test_degenerate_gives_p_k(self):
        for k in range(4):
            gamma_degen = np.zeros((1, 4))
            gamma_degen[0, k] = 1.0
            pd = pd_from_joint(gamma_degen, TRUE_P)
            assert abs(pd[0] - TRUE_P[k]) < 1e-12


class TestPdFromProduct:
    def test_output_shape(self):
        gamma = _make_gamma(60)
        q_pm = compute_product_marginals(gamma)
        pd_pm = pd_from_product(q_pm, TRUE_P)
        assert pd_pm.shape == (60,)

    def test_bounded(self):
        gamma = _make_gamma(60)
        q_pm = compute_product_marginals(gamma)
        pd_pm = pd_from_product(q_pm, TRUE_P)
        assert np.all(pd_pm >= TRUE_P.min() - 1e-10)
        assert np.all(pd_pm <= TRUE_P.max() + 1e-10)

    def test_equals_joint_pd_when_independent(self):
        """When gamma is factorized, pd_from_joint and pd_from_product must agree."""
        gamma_indep = _make_independent_gamma(100, seed=22)
        q_pm = compute_product_marginals(gamma_indep)
        pd_j = pd_from_joint(gamma_indep, TRUE_P)
        pd_p = pd_from_product(q_pm, TRUE_P)
        np.testing.assert_allclose(pd_j, pd_p, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for summarize_pm_metrics
# ---------------------------------------------------------------------------

class TestSummarizePmMetrics:
    def setup_method(self):
        self.gamma = _make_gamma(200, seed=88)
        self.p_vec = TRUE_P.copy()

    def test_returns_correct_types(self):
        metrics, q_pm, kl_vals, pd_joint, pd_pm = summarize_pm_metrics(
            self.gamma, self.p_vec
        )
        assert isinstance(metrics, ProductMarginalsMetrics)

    def test_q_pm_shape(self):
        _, q_pm, _, _, _ = summarize_pm_metrics(self.gamma, self.p_vec)
        assert q_pm.shape == (200, 4)

    def test_kl_values_shape(self):
        _, _, kl_vals, _, _ = summarize_pm_metrics(self.gamma, self.p_vec)
        assert kl_vals.shape == (200,)

    def test_pd_shapes(self):
        _, _, _, pd_joint, pd_pm = summarize_pm_metrics(self.gamma, self.p_vec)
        assert pd_joint.shape == (200,)
        assert pd_pm.shape == (200,)

    def test_kl_avg_nonnegative(self):
        metrics, _, _, _, _ = summarize_pm_metrics(self.gamma, self.p_vec)
        assert metrics.kl_avg >= 0

    def test_mae_pd_pm_nonnegative(self):
        metrics, _, _, _, _ = summarize_pm_metrics(self.gamma, self.p_vec)
        assert metrics.mae_pd_pm >= 0

    def test_zero_kl_when_independent(self):
        """When gamma is factorized, KL must be ~0."""
        gamma_indep = _make_independent_gamma(200, seed=55)
        metrics, _, _, _, _ = summarize_pm_metrics(gamma_indep, self.p_vec)
        assert metrics.kl_avg < 1e-8

    def test_zero_mae_pd_when_independent(self):
        """When gamma is factorized, PD distortion must be ~0."""
        gamma_indep = _make_independent_gamma(200, seed=66)
        metrics, _, _, _, _ = summarize_pm_metrics(gamma_indep, self.p_vec)
        assert metrics.mae_pd_pm < 1e-10

    def test_input_validation_wrong_columns(self):
        """Must raise AssertionError with wrong gamma columns."""
        gamma_bad = np.ones((10, 3)) / 3
        with pytest.raises(AssertionError):
            summarize_pm_metrics(gamma_bad, self.p_vec)

    def test_input_validation_rows_not_summing_to_one(self):
        """Must raise AssertionError if gamma rows don't sum to 1."""
        gamma_bad = np.ones((10, 4))  # rows sum to 4, not 1
        with pytest.raises(AssertionError):
            summarize_pm_metrics(gamma_bad, self.p_vec)
