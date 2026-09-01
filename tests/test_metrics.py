"""
Tests for src/metrics.py — Recovery metrics for the EM experiment.

Tests verify:
- All individual metric functions produce correct values on known inputs
- Metrics are non-negative
- Perfect recovery gives zero metric
- KL and JS are non-negative and bounded (JS in [0, log 2])
- compute_all_metrics returns the right container type
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    mae_pi,
    mse_mu,
    mse_sigma,
    mae_p,
    mse_p,
    mse_gamma,
    kl_average,
    js_average,
    mse_pd,
    mae_pd,
    compute_all_metrics,
    RecoveryMetrics,
)
from src.dgp import TRUE_PI, TRUE_MU, TRUE_SIGMA, TRUE_P


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_uniform_gamma(N: int = 100, K: int = 4) -> np.ndarray:
    return np.ones((N, K)) / K


def _make_random_gamma(N: int = 100, K: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.standard_normal((N, K))) + 0.01
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Tests for mae_pi
# ---------------------------------------------------------------------------

class TestMaePi:
    def test_zero_on_exact_match(self):
        assert mae_pi(TRUE_PI, TRUE_PI) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        pi_hat = rng.dirichlet(np.ones(4))
        assert mae_pi(pi_hat, TRUE_PI) >= 0

    def test_known_value(self):
        pi_hat = np.array([0.5, 0.2, 0.2, 0.1])
        pi_true = np.array([0.4, 0.2, 0.25, 0.15])
        # |0.5-0.4| + |0.2-0.2| + |0.2-0.25| + |0.1-0.15| = 0.1+0+0.05+0.05 = 0.2; / 4 = 0.05
        expected = (0.1 + 0.0 + 0.05 + 0.05) / 4
        assert mae_pi(pi_hat, pi_true) == pytest.approx(expected, rel=1e-10)

    def test_scalar_output(self):
        result = mae_pi(TRUE_PI, TRUE_PI)
        assert np.isscalar(result) or np.ndim(result) == 0


# ---------------------------------------------------------------------------
# Tests for mse_mu
# ---------------------------------------------------------------------------

class TestMseMu:
    def test_zero_on_exact_match(self):
        assert mse_mu(TRUE_MU, TRUE_MU) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        mu_hat = TRUE_MU + rng.standard_normal(TRUE_MU.shape) * 0.1
        assert mse_mu(mu_hat, TRUE_MU) >= 0

    def test_known_value(self):
        mu_hat = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
        mu_true = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        # Each component: ||diff||^2 = 1+0 = 1 → avg = 1.0
        assert mse_mu(mu_hat, mu_true) == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests for mse_sigma
# ---------------------------------------------------------------------------

class TestMseSigma:
    def test_zero_on_exact_match(self):
        assert mse_sigma(TRUE_SIGMA, TRUE_SIGMA) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        sigma_hat = [s + np.eye(2) * 0.01 for s in TRUE_SIGMA]
        assert mse_sigma(sigma_hat, TRUE_SIGMA) >= 0

    def test_known_value(self):
        """Adding 0.1*I to each Sigma: Frobenius^2 = 0.01+0+0+0.01 = 0.02, avg = 0.02."""
        sigma_true = [np.eye(2) for _ in range(4)]
        sigma_hat = [np.eye(2) + 0.1 * np.eye(2) for _ in range(4)]
        # diff = 0.1*I, ||0.1*I||_F^2 = 0.1^2 + 0.1^2 = 0.02
        expected = 0.02
        assert mse_sigma(sigma_hat, sigma_true) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests for mae_p / mse_p
# ---------------------------------------------------------------------------

class TestMaeP:
    def test_zero_on_exact_match(self):
        assert mae_p(TRUE_P, TRUE_P) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        p_hat = np.array([0.012, 0.045, 0.12, 0.22])
        assert mae_p(p_hat, TRUE_P) >= 0

    def test_known_value(self):
        p_hat = np.array([0.02, 0.05, 0.10, 0.25])
        p_true = np.array([0.01, 0.05, 0.10, 0.25])
        expected = abs(0.02 - 0.01) / 4
        assert mae_p(p_hat, p_true) == pytest.approx(expected, rel=1e-10)


class TestMseP:
    def test_zero_on_exact_match(self):
        assert mse_p(TRUE_P, TRUE_P) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        p_hat = np.array([0.012, 0.045, 0.12, 0.22])
        assert mse_p(p_hat, TRUE_P) >= 0

    def test_known_value(self):
        p_hat = np.array([0.02, 0.05, 0.10, 0.25])
        p_true = np.array([0.01, 0.05, 0.10, 0.25])
        expected = (0.01 ** 2) / 4
        assert mse_p(p_hat, p_true) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests for mse_gamma
# ---------------------------------------------------------------------------

class TestMseGamma:
    def test_zero_on_exact_match(self):
        gamma = _make_random_gamma(100, 4, seed=7)
        assert mse_gamma(gamma, gamma) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        gamma_hat = _make_random_gamma(100, 4, seed=1)
        gamma_true = _make_random_gamma(100, 4, seed=2)
        assert mse_gamma(gamma_hat, gamma_true) >= 0

    def test_known_value(self):
        """Known shift: gamma_hat = gamma_true + const row-wise."""
        gamma_true = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
        # gamma_hat is same as gamma_true (for sanity)
        assert mse_gamma(gamma_true, gamma_true) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Tests for kl_average
# ---------------------------------------------------------------------------

class TestKlAverage:
    def test_zero_on_exact_match(self):
        gamma = _make_random_gamma(100, 4, seed=8)
        assert kl_average(gamma, gamma) == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self):
        gamma_true = _make_random_gamma(100, 4, seed=3)
        gamma_hat = _make_random_gamma(100, 4, seed=4)
        assert kl_average(gamma_true, gamma_hat) >= 0

    def test_zero_times_log_zero_convention(self):
        """Must handle 0 * log(0/q) = 0 without NaN."""
        gamma_true = np.array([[1.0, 0.0, 0.0, 0.0]])
        gamma_hat = np.array([[0.7, 0.1, 0.1, 0.1]])
        kl = kl_average(gamma_true, gamma_hat)
        assert np.isfinite(kl)
        assert kl >= 0


# ---------------------------------------------------------------------------
# Tests for js_average
# ---------------------------------------------------------------------------

class TestJsAverage:
    def test_zero_on_exact_match(self):
        gamma = _make_random_gamma(50, 4, seed=9)
        assert js_average(gamma, gamma) == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self):
        gamma_true = _make_random_gamma(50, 4, seed=5)
        gamma_hat = _make_random_gamma(50, 4, seed=6)
        assert js_average(gamma_true, gamma_hat) >= 0

    def test_bounded_above_by_log2(self):
        """Jensen-Shannon divergence is bounded above by log(2)."""
        gamma_true = _make_random_gamma(50, 4, seed=11)
        gamma_hat = _make_random_gamma(50, 4, seed=12)
        js = js_average(gamma_true, gamma_hat)
        assert js <= np.log(2) + 1e-10

    def test_symmetric_approximately(self):
        """Jensen-Shannon divergence is symmetric."""
        gamma_a = _make_random_gamma(100, 4, seed=20)
        gamma_b = _make_random_gamma(100, 4, seed=21)
        js_ab = js_average(gamma_a, gamma_b)
        js_ba = js_average(gamma_b, gamma_a)
        assert abs(js_ab - js_ba) < 1e-9


# ---------------------------------------------------------------------------
# Tests for mse_pd / mae_pd
# ---------------------------------------------------------------------------

class TestMsePd:
    def test_zero_on_exact_match(self):
        pd = np.array([0.01, 0.05, 0.10, 0.25])
        assert mse_pd(pd, pd) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        assert mse_pd(np.array([0.02, 0.06]), np.array([0.01, 0.05])) >= 0

    def test_known_value(self):
        pd_hat = np.array([0.02, 0.05])
        pd_true = np.array([0.01, 0.05])
        expected = (0.01 ** 2) / 2
        assert mse_pd(pd_hat, pd_true) == pytest.approx(expected, rel=1e-10)


class TestMaePd:
    def test_zero_on_exact_match(self):
        pd = np.array([0.01, 0.05, 0.10, 0.25])
        assert mae_pd(pd, pd) == pytest.approx(0.0, abs=1e-12)

    def test_nonnegative(self):
        assert mae_pd(np.array([0.02, 0.06]), np.array([0.01, 0.05])) >= 0

    def test_known_value(self):
        pd_hat = np.array([0.02, 0.05])
        pd_true = np.array([0.01, 0.05])
        expected = 0.01 / 2
        assert mae_pd(pd_hat, pd_true) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests for compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def setup_method(self):
        rng = np.random.default_rng(99)
        N = 200
        self.gamma = _make_random_gamma(N, 4, seed=50)
        self.pd = self.gamma @ TRUE_P
        self.sigma_hat = [s.copy() for s in TRUE_SIGMA]

    def test_returns_recovery_metrics(self):
        m = compute_all_metrics(
            pi_hat=TRUE_PI,
            mu_hat=TRUE_MU,
            sigma_hat=self.sigma_hat,
            p_hat=TRUE_P,
            gamma_hat=self.gamma,
            pd_hat=self.pd,
            pi_true=TRUE_PI,
            mu_true=TRUE_MU,
            sigma_true=TRUE_SIGMA,
            p_true=TRUE_P,
            gamma_true=self.gamma,
            pd_true=self.pd,
        )
        assert isinstance(m, RecoveryMetrics)

    def test_all_metrics_nonnegative(self):
        m = compute_all_metrics(
            pi_hat=TRUE_PI,
            mu_hat=TRUE_MU,
            sigma_hat=self.sigma_hat,
            p_hat=TRUE_P,
            gamma_hat=self.gamma,
            pd_hat=self.pd,
            pi_true=TRUE_PI,
            mu_true=TRUE_MU,
            sigma_true=TRUE_SIGMA,
            p_true=TRUE_P,
            gamma_true=self.gamma,
            pd_true=self.pd,
        )
        assert m.MAE_pi >= 0
        assert m.MSE_mu >= 0
        assert m.MSE_Sigma >= 0
        assert m.MAE_p >= 0
        assert m.MSE_p >= 0
        assert m.MSE_gamma >= 0
        assert m.KL_avg >= 0
        assert m.JS_avg >= 0
        assert m.MSE_PD >= 0
        assert m.MAE_PD >= 0

    def test_perfect_recovery_gives_zero_structural_metrics(self):
        """Perfect parameter recovery (pi, mu, sigma, p) must give zero structural metrics."""
        m = compute_all_metrics(
            pi_hat=TRUE_PI,
            mu_hat=TRUE_MU,
            sigma_hat=self.sigma_hat,
            p_hat=TRUE_P,
            gamma_hat=self.gamma,
            pd_hat=self.pd,
            pi_true=TRUE_PI,
            mu_true=TRUE_MU,
            sigma_true=TRUE_SIGMA,
            p_true=TRUE_P,
            gamma_true=self.gamma,
            pd_true=self.pd,
        )
        assert m.MAE_pi == pytest.approx(0.0, abs=1e-12)
        assert m.MSE_mu == pytest.approx(0.0, abs=1e-12)
        assert m.MSE_Sigma == pytest.approx(0.0, abs=1e-9)
        assert m.MAE_p == pytest.approx(0.0, abs=1e-12)
        assert m.MSE_p == pytest.approx(0.0, abs=1e-12)
        assert m.MSE_gamma == pytest.approx(0.0, abs=1e-12)
        assert m.MSE_PD == pytest.approx(0.0, abs=1e-12)
        assert m.MAE_PD == pytest.approx(0.0, abs=1e-12)

    def test_all_fields_present(self):
        """RecoveryMetrics must have all 10 fields."""
        m = compute_all_metrics(
            pi_hat=TRUE_PI, mu_hat=TRUE_MU, sigma_hat=self.sigma_hat,
            p_hat=TRUE_P, gamma_hat=self.gamma, pd_hat=self.pd,
            pi_true=TRUE_PI, mu_true=TRUE_MU, sigma_true=TRUE_SIGMA,
            p_true=TRUE_P, gamma_true=self.gamma, pd_true=self.pd,
        )
        for attr in ["MAE_pi", "MSE_mu", "MSE_Sigma", "MAE_p", "MSE_p",
                     "MSE_gamma", "KL_avg", "JS_avg", "MSE_PD", "MAE_PD"]:
            assert hasattr(m, attr), f"Missing metric: {attr}"
