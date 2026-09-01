"""
Posterior belief and borrower-level PD reconstruction.

After EM fitting and label alignment, this module computes:
  1. Score-based predictive posterior beliefs:
       hat_gamma_ik^X = hat_pi_k phi_2(X_i; hat_mu_k, hat_Sigma_k) /
                        sum_j hat_pi_j phi_2(X_i; hat_mu_j, hat_Sigma_j)
     NOTE: D_i is NOT used here — this is a score-only posterior.

  2. Estimated borrower-level PDs:
       hat_PD_i = sum_k hat_p_k hat_gamma_ik^X
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp  # Context7 confirmed

from src.em_estimator import _mvn_logpdf_batch, P_EPS


def compute_score_posteriors(
    X: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: list[np.ndarray],
) -> np.ndarray:
    """
    Compute score-based posterior beliefs hat_gamma_i^X using scores only.

    hat_gamma_ik^X = hat_pi_k phi_2(X_i; hat_mu_k, hat_Sigma_k) /
                     sum_j hat_pi_j phi_2(X_i; hat_mu_j, hat_Sigma_j)

    D_i is NOT included. This is the predictive posterior conditioned on
    scores only, distinct from EM responsibilities tau_ik which condition
    on both scores and default.

    Parameters
    ----------
    X     : (N, 2) score matrix
    pi    : (K,)   mixture weights (aligned)
    mu    : (K, D) component means (aligned)
    sigma : K-list of (D, D) covariance matrices (aligned)

    Returns
    -------
    gamma : (N, K) score-based posterior matrix; rows sum to 1.
    """
    N = X.shape[0]
    K = pi.shape[0]

    # Log unnormalized: log pi_k + log phi_2(X_i; mu_k, Sigma_k)
    log_unnorm = np.zeros((N, K), dtype=float)
    for k in range(K):
        log_unnorm[:, k] = np.log(pi[k]) + _mvn_logpdf_batch(X, mu[k], sigma[k])

    # Normalize with log-sum-exp
    log_norm = logsumexp(log_unnorm, axis=1, keepdims=True)
    gamma = np.exp(log_unnorm - log_norm)

    # Validate: rows must sum to 1
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-9), \
        "Score-based posterior rows do not sum to 1"
    return gamma


def compute_borrower_pd(
    gamma: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Compute estimated borrower-level probability of default.

    hat_PD_i = sum_k hat_p_k hat_gamma_ik^X

    Parameters
    ----------
    gamma : (N, K) score-based posterior matrix
    p     : (K,)   profile-specific default probabilities

    Returns
    -------
    PD : (N,) estimated borrower-level PDs
    """
    assert gamma.shape[1] == p.shape[0], \
        f"Dimension mismatch: gamma has {gamma.shape[1]} columns but p has {p.shape[0]} elements"
    PD = gamma @ p  # (N,)
    return PD
