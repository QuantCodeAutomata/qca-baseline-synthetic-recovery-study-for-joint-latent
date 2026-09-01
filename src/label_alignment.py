"""
Label alignment for EM estimates against true parameters.

Because latent class labels are permutation-invariant, estimated parameters
must be aligned to true labels before computing recovery metrics.

Alignment criterion: find the permutation sigma that minimizes
  sum_k || hat_mu_{sigma(k)} - mu_k* ||^2

Solved via the Hungarian assignment algorithm (minimizing total squared
mean distance across the K=4 component pairs).

# Using scipy.optimize.linear_sum_assignment — Context7 confirmed
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # Context7 confirmed


def build_cost_matrix(
    mu_hat: np.ndarray,
    mu_true: np.ndarray,
) -> np.ndarray:
    """
    Build K x K cost matrix of squared Euclidean distances between estimated
    and true component means.

    cost[j, k] = || hat_mu_j - mu_k* ||^2

    Parameters
    ----------
    mu_hat  : (K, D) estimated means
    mu_true : (K, D) true means

    Returns
    -------
    cost : (K, K) matrix where cost[j, k] is the cost of assigning estimated
           component j to true component k.
    """
    K = mu_hat.shape[0]
    cost = np.zeros((K, K), dtype=float)
    for j in range(K):
        for k in range(K):
            diff = mu_hat[j] - mu_true[k]
            cost[j, k] = np.dot(diff, diff)
    return cost


def find_optimal_permutation(
    mu_hat: np.ndarray,
    mu_true: np.ndarray,
) -> np.ndarray:
    """
    Find the permutation of estimated component indices that best aligns
    estimated means to true means (minimizing total squared mean distance).

    Parameters
    ----------
    mu_hat  : (K, D) estimated means
    mu_true : (K, D) true means

    Returns
    -------
    perm : (K,) integer array such that perm[k] is the estimated index
           that maps to true index k.  That is, hat_mu[perm[k]] is the
           estimated component aligned to true component k.
    """
    cost = build_cost_matrix(mu_hat, mu_true)
    # linear_sum_assignment returns (row_ind, col_ind): assigns estimated row
    # to true column — row_ind[i] is estimated index, col_ind[i] is true index
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the permutation: perm[true_k] = estimated_j
    K = mu_hat.shape[0]
    perm = np.zeros(K, dtype=int)
    for est_j, true_k in zip(row_ind, col_ind):
        perm[true_k] = est_j

    return perm


def align_estimates(
    pi_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: list[np.ndarray],
    p_hat: np.ndarray,
    tau: np.ndarray,
    mu_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Reorder all estimated class-indexed objects to align with true labels.

    Aligned objects satisfy: aligned_hat_mu[k] is closest to mu_true[k].

    Parameters
    ----------
    pi_hat    : (K,)   estimated mixture weights
    mu_hat    : (K, D) estimated means
    sigma_hat : K-list of (D, D) estimated covariances
    p_hat     : (K,)   estimated default probabilities
    tau       : (N, K) EM responsibilities
    mu_true   : (K, D) true means for alignment reference

    Returns
    -------
    pi_aligned    : (K,)
    mu_aligned    : (K, D)
    sigma_aligned : K-list of (D, D)
    p_aligned     : (K,)
    tau_aligned   : (N, K)
    """
    perm = find_optimal_permutation(mu_hat, mu_true)

    pi_aligned = pi_hat[perm]
    mu_aligned = mu_hat[perm]
    sigma_aligned = [sigma_hat[perm[k]] for k in range(len(perm))]
    p_aligned = p_hat[perm]
    tau_aligned = tau[:, perm]

    return pi_aligned, mu_aligned, sigma_aligned, p_aligned, tau_aligned
