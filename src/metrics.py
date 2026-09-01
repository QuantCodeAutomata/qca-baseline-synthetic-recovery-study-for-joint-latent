"""
Recovery metrics for the joint latent-state EM study.

Computes all metrics exactly as specified in the paper methodology:

  MAE_pi    = (1/K) sum_k |hat_pi_k - pi_k*|
  MSE_mu    = (1/K) sum_k ||hat_mu_k - mu_k*||^2
  MSE_Sigma = (1/K) sum_k ||hat_Sigma_k - Sigma_k*||_F^2
  MAE_p     = (1/K) sum_k |hat_p_k - p_k*|
  MSE_p     = (1/K) sum_k (hat_p_k - p_k*)^2
  MSE_gamma = (1/n) sum_i ||hat_gamma_i^X - gamma_i^{X,*}||^2
  KL_avg    = (1/n) sum_i D_KL(gamma_i^{X,*} || hat_gamma_i^X)
  JS_avg    = (1/n) sum_i JSD(gamma_i^{X,*}, hat_gamma_i^X)
  MSE_PD    = (1/n) sum_i (hat_PD_i - PD_i*)^2
  MAE_PD    = (1/n) sum_i |hat_PD_i - PD_i*|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------

LOG_CLIP: float = 1e-300  # prevent log(0) in KL/JS; use 0*log(0/q)=0 convention


def _safe_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    KL divergence D_KL(p || q) per borrower row, applying 0*log(0/q) = 0 convention.

    Parameters
    ----------
    p, q : (N, K) probability matrices; rows sum to 1.

    Returns
    -------
    kl : (N,) per-borrower KL divergences
    """
    # Clip q to avoid division by zero; apply convention 0*log(0/q)=0 via masking
    q_safe = np.where(q > 0, q, 1.0)   # replace 0 with 1 (safe in denominator)
    p_safe = np.where(p > 0, p, 1.0)   # replace 0 with 1 (will be masked out)
    log_ratio = np.log(p_safe) - np.log(q_safe)
    kl = np.where(p > 0, p * log_ratio, 0.0).sum(axis=1)
    # KL must be non-negative
    kl = np.maximum(kl, 0.0)
    return kl


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Jensen-Shannon divergence JSD(p, q) per borrower row.

    JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q).

    Satisfies 0 <= JSD <= log(2).

    Parameters
    ----------
    p, q : (N, K) probability matrices; rows sum to 1.

    Returns
    -------
    js : (N,) per-borrower JS divergences
    """
    m = 0.5 * (p + q)
    js = 0.5 * _safe_kl(p, m) + 0.5 * _safe_kl(q, m)
    return js


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecoveryMetrics:
    """Container for all recovery metrics from the experiment."""
    MAE_pi: float
    MSE_mu: float
    MSE_Sigma: float
    MAE_p: float
    MSE_p: float
    MSE_gamma: float
    KL_avg: float      # average D_KL(true || estimated)
    JS_avg: float      # average Jensen-Shannon divergence
    MSE_PD: float
    MAE_PD: float


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def mae_pi(pi_hat: np.ndarray, pi_true: np.ndarray) -> float:
    """MAE_pi = (1/K) sum_k |hat_pi_k - pi_k*|"""
    return float(np.mean(np.abs(pi_hat - pi_true)))


def mse_mu(mu_hat: np.ndarray, mu_true: np.ndarray) -> float:
    """MSE_mu = (1/K) sum_k ||hat_mu_k - mu_k*||^2"""
    diff = mu_hat - mu_true  # (K, D)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def mse_sigma(
    sigma_hat: list[np.ndarray],
    sigma_true: list[np.ndarray],
) -> float:
    """MSE_Sigma = (1/K) sum_k ||hat_Sigma_k - Sigma_k*||_F^2"""
    K = len(sigma_hat)
    total = 0.0
    for k in range(K):
        diff = sigma_hat[k] - sigma_true[k]
        total += np.sum(diff ** 2)  # Frobenius norm squared
    return total / K


def mae_p(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    """MAE_p = (1/K) sum_k |hat_p_k - p_k*|"""
    return float(np.mean(np.abs(p_hat - p_true)))


def mse_p(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    """MSE_p = (1/K) sum_k (hat_p_k - p_k*)^2"""
    return float(np.mean((p_hat - p_true) ** 2))


def mse_gamma(gamma_hat: np.ndarray, gamma_true: np.ndarray) -> float:
    """MSE_gamma = (1/n) sum_i ||hat_gamma_i^X - gamma_i^{X,*}||^2"""
    diff = gamma_hat - gamma_true  # (N, K)
    return float(np.mean(np.sum(diff ** 2, axis=1)))


def kl_average(gamma_true: np.ndarray, gamma_hat: np.ndarray) -> float:
    """
    Average borrower-level KL divergence from true to estimated posterior.

    KL_avg = (1/n) sum_i D_KL(gamma_i^{X,*} || hat_gamma_i^X)
    """
    kl = _safe_kl(gamma_true, gamma_hat)
    return float(kl.mean())


def js_average(gamma_true: np.ndarray, gamma_hat: np.ndarray) -> float:
    """
    Average borrower-level Jensen-Shannon divergence.

    JS_avg = (1/n) sum_i JSD(gamma_i^{X,*}, hat_gamma_i^X)
    """
    js = _js_divergence(gamma_true, gamma_hat)
    return float(js.mean())


def mse_pd(pd_hat: np.ndarray, pd_true: np.ndarray) -> float:
    """MSE_PD = (1/n) sum_i (hat_PD_i - PD_i*)^2"""
    return float(np.mean((pd_hat - pd_true) ** 2))


def mae_pd(pd_hat: np.ndarray, pd_true: np.ndarray) -> float:
    """MAE_PD = (1/n) sum_i |hat_PD_i - PD_i*|"""
    return float(np.mean(np.abs(pd_hat - pd_true)))


# ---------------------------------------------------------------------------
# Composite metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pi_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: list[np.ndarray],
    p_hat: np.ndarray,
    gamma_hat: np.ndarray,
    pd_hat: np.ndarray,
    pi_true: np.ndarray,
    mu_true: np.ndarray,
    sigma_true: list[np.ndarray],
    p_true: np.ndarray,
    gamma_true: np.ndarray,
    pd_true: np.ndarray,
) -> RecoveryMetrics:
    """
    Compute all ten recovery metrics from the paper.

    Parameters
    ----------
    pi_hat, mu_hat, sigma_hat, p_hat  : aligned estimated parameters
    gamma_hat                          : (N, K) estimated score-based posteriors
    pd_hat                             : (N,)   estimated borrower-level PDs
    pi_true, mu_true, sigma_true, p_true : true parameters
    gamma_true                          : (N, K) true score-based posteriors
    pd_true                             : (N,)   true borrower-level PDs

    Returns
    -------
    RecoveryMetrics dataclass with all ten metrics.
    """
    return RecoveryMetrics(
        MAE_pi=mae_pi(pi_hat, pi_true),
        MSE_mu=mse_mu(mu_hat, mu_true),
        MSE_Sigma=mse_sigma(sigma_hat, sigma_true),
        MAE_p=mae_p(p_hat, p_true),
        MSE_p=mse_p(p_hat, p_true),
        MSE_gamma=mse_gamma(gamma_hat, gamma_true),
        KL_avg=kl_average(gamma_true, gamma_hat),
        JS_avg=js_average(gamma_true, gamma_hat),
        MSE_PD=mse_pd(pd_hat, pd_true),
        MAE_PD=mae_pd(pd_hat, pd_true),
    )
