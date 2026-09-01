"""
Tests for src/label_alignment.py — Label alignment via Hungarian algorithm.

Tests verify:
- Cost matrix construction
- Permutation correctness (identity on already-aligned data, correct swap)
- align_estimates reorders all objects consistently
- Consistency after alignment: aligned mu matches true mu best
"""

from __future__ import annotations

import numpy as np
import pytest

from src.label_alignment import (
    build_cost_matrix,
    find_optimal_permutation,
    align_estimates,
)
from src.dgp import TRUE_MU


# ---------------------------------------------------------------------------
# Tests for build_cost_matrix
# ---------------------------------------------------------------------------

class TestBuildCostMatrix:
    def test_shape(self):
        """Cost matrix must be (K, K)."""
        K = 4
        mu_hat = np.random.default_rng(0).standard_normal((K, 2))
        mu_true = np.random.default_rng(1).standard_normal((K, 2))
        cost = build_cost_matrix(mu_hat, mu_true)
        assert cost.shape == (K, K)

    def test_nonnegative(self):
        """Cost entries must be non-negative (squared distances)."""
        mu_hat = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        mu_true = np.array([[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5]])
        cost = build_cost_matrix(mu_hat, mu_true)
        assert np.all(cost >= 0)

    def test_zero_diagonal_when_identical(self):
        """When mu_hat == mu_true, diagonal must be zero."""
        mu = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        cost = build_cost_matrix(mu, mu)
        np.testing.assert_allclose(np.diag(cost), np.zeros(4), atol=1e-12)

    def test_correct_value(self):
        """Spot-check a specific cost value."""
        mu_hat = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        mu_true = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cost = build_cost_matrix(mu_hat, mu_true)
        assert abs(cost[0, 0] - 1.0) < 1e-12  # (1-2)^2 + 0^2 = 1


# ---------------------------------------------------------------------------
# Tests for find_optimal_permutation
# ---------------------------------------------------------------------------

class TestFindOptimalPermutation:
    def test_identity_when_aligned(self):
        """When mu_hat == mu_true, permutation must be identity."""
        mu = np.array([[1.8, 0.3], [1.8, 1.8], [-0.8, 0.3], [-0.8, 1.8]])
        perm = find_optimal_permutation(mu, mu)
        np.testing.assert_array_equal(perm, np.arange(4))

    def test_single_swap(self):
        """Swapped components 0 and 1 must be correctly identified and reversed."""
        mu_true = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
        # Swap components 0 and 1
        mu_hat = np.array([[0.0, 2.0], [2.0, 0.0], [-2.0, 0.0], [0.0, -2.0]])
        perm = find_optimal_permutation(mu_hat, mu_true)
        # perm[k] = estimated index aligned to true k
        # True 0 should be aligned with estimated 1, and vice versa
        assert perm[0] == 1
        assert perm[1] == 0
        assert perm[2] == 2
        assert perm[3] == 3

    def test_perm_is_valid_permutation(self):
        """Returned permutation must be a valid permutation of {0,1,2,3}."""
        rng = np.random.default_rng(5)
        mu_hat = rng.standard_normal((4, 2))
        mu_true = rng.standard_normal((4, 2))
        perm = find_optimal_permutation(mu_hat, mu_true)
        assert sorted(perm.tolist()) == list(range(4))

    def test_permutation_reduces_cost(self):
        """The permutation must minimize total squared mean distance."""
        rng = np.random.default_rng(99)
        mu_hat = rng.standard_normal((4, 2))
        mu_true = rng.standard_normal((4, 2))
        perm = find_optimal_permutation(mu_hat, mu_true)

        # Cost with optimal permutation
        optimal_cost = sum(
            np.sum((mu_hat[perm[k]] - mu_true[k]) ** 2) for k in range(4)
        )
        # Cost with identity (no permutation)
        identity_cost = sum(
            np.sum((mu_hat[k] - mu_true[k]) ** 2) for k in range(4)
        )
        assert optimal_cost <= identity_cost + 1e-10


# ---------------------------------------------------------------------------
# Tests for align_estimates
# ---------------------------------------------------------------------------

class TestAlignEstimates:
    def setup_method(self):
        """Set up mock EM estimates that are a permutation of true values."""
        self.mu_true = TRUE_MU.copy()
        # Rotate: component 3 -> 0, 0 -> 1, 1 -> 2, 2 -> 3
        rotation = [3, 0, 1, 2]
        self.pi_hat_unaligned = np.array([0.15, 0.40, 0.20, 0.25])  # rotated
        self.mu_hat_unaligned = self.mu_true[rotation]
        self.sigma_hat_unaligned = [
            np.array([[0.45, 0.10], [0.10, 0.45]]),
            np.array([[0.40, 0.10], [0.10, 0.40]]),
            np.array([[0.40, 0.10], [0.10, 0.40]]),
            np.array([[0.45, 0.10], [0.10, 0.45]]),
        ]
        self.p_hat_unaligned = np.array([0.25, 0.01, 0.05, 0.10])  # rotated
        rng = np.random.default_rng(0)
        tau_raw = np.abs(rng.standard_normal((100, 4)))
        self.tau_raw = tau_raw / tau_raw.sum(axis=1, keepdims=True)

    def test_mu_alignment_correct(self):
        """After alignment, mu should match true mu."""
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            self.pi_hat_unaligned,
            self.mu_hat_unaligned,
            self.sigma_hat_unaligned,
            self.p_hat_unaligned,
            self.tau_raw,
            self.mu_true,
        )
        np.testing.assert_allclose(mu_a, self.mu_true, atol=1e-10)

    def test_pi_sums_to_one_after_alignment(self):
        """pi must still sum to 1 after alignment."""
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            self.pi_hat_unaligned,
            self.mu_hat_unaligned,
            self.sigma_hat_unaligned,
            self.p_hat_unaligned,
            self.tau_raw,
            self.mu_true,
        )
        assert abs(pi_a.sum() - 1.0) < 1e-10

    def test_tau_rows_sum_to_one_after_alignment(self):
        """tau rows must still sum to 1 after alignment."""
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            self.pi_hat_unaligned,
            self.mu_hat_unaligned,
            self.sigma_hat_unaligned,
            self.p_hat_unaligned,
            self.tau_raw,
            self.mu_true,
        )
        np.testing.assert_allclose(tau_a.sum(axis=1), np.ones(100), atol=1e-10)

    def test_sigma_count_preserved(self):
        """Number of sigma matrices must equal K=4 after alignment."""
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            self.pi_hat_unaligned,
            self.mu_hat_unaligned,
            self.sigma_hat_unaligned,
            self.p_hat_unaligned,
            self.tau_raw,
            self.mu_true,
        )
        assert len(sigma_a) == 4

    def test_p_values_permuted_correctly(self):
        """p must be reordered to match the same permutation as mu."""
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            self.pi_hat_unaligned,
            self.mu_hat_unaligned,
            self.sigma_hat_unaligned,
            self.p_hat_unaligned,
            self.tau_raw,
            self.mu_true,
        )
        # After alignment, p should match the true ordering (0.01, 0.05, 0.10, 0.25)
        np.testing.assert_allclose(
            p_a, np.array([0.01, 0.05, 0.10, 0.25]), atol=1e-10
        )

    def test_identity_permutation_unchanged(self):
        """When already aligned, objects must be returned unchanged."""
        pi_original = np.array([0.40, 0.20, 0.25, 0.15])
        sigma_original = [
            np.array([[0.40, 0.10], [0.10, 0.40]]),
            np.array([[0.40, 0.10], [0.10, 0.40]]),
            np.array([[0.45, 0.10], [0.10, 0.45]]),
            np.array([[0.45, 0.10], [0.10, 0.45]]),
        ]
        p_original = np.array([0.01, 0.05, 0.10, 0.25])
        tau_original = self.tau_raw.copy()
        pi_a, mu_a, sigma_a, p_a, tau_a = align_estimates(
            pi_original,
            self.mu_true,
            sigma_original,
            p_original,
            tau_original,
            self.mu_true,
        )
        np.testing.assert_allclose(pi_a, pi_original, atol=1e-10)
        np.testing.assert_allclose(mu_a, self.mu_true, atol=1e-10)
        np.testing.assert_allclose(p_a, p_original, atol=1e-10)
        np.testing.assert_allclose(tau_a, tau_original, atol=1e-10)
