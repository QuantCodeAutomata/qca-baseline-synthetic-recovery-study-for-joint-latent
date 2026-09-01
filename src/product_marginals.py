"""
Product-of-marginals approximation analysis for borrower-level latent posteriors.

Analyzes information loss incurred when the joint posterior over latent
creditworthiness (Z_R) and fragility (Z_F) states is approximated by the
product of its marginal posteriors.

Profile ordering (fixed):
  k=0 -> (Z_R=0, Z_F=0)
  k=1 -> (Z_R=0, Z_F=1)
  k=2 -> (Z_R=1, Z_F=0)
  k=3 -> (Z_R=1, Z_F=1)

Custom — Context7 found no library equivalent for this exact posterior
factorization analysis in a 2-dimensional binary latent space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Posterior marginals and factorized approximation
# ---------------------------------------------------------------------------

def compute_posterior_marginals(
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal posteriors over Z_R and Z_F from the 4-state joint posterior.

    hat_p_R,i^X(0) = gamma_i1 + gamma_i2  [profiles with Z_R=0]
    hat_p_R,i^X(1) = gamma_i3 + gamma_i4  [profiles with Z_R=1]
    hat_p_F,i^X(0) = gamma_i1 + gamma_i3  [profiles with Z_F=0]
    hat_p_F,i^X(1) = gamma_i2 + gamma_i4  [profiles with Z_F=1]

    Parameters
    ----------
    gamma : (N, 4) joint posterior over profiles ordered as
            [(0,0), (0,1), (1,0), (1,1)]

    Returns
    -------
    marg_R : (N, 2) marginal over Z_R: columns are [P(Z_R=0), P(Z_R=1)]
    marg_F : (N, 2) marginal over Z_F: columns are [P(Z_F=0), P(Z_F=1)]
    """
    assert gamma.shape[1] == 4, "gamma must have 4 columns (one per profile)"

    # Z_R marginal
    marg_R_0 = gamma[:, 0] + gamma[:, 1]  # Z_R=0: profiles (0,0) + (0,1)
    marg_R_1 = gamma[:, 2] + gamma[:, 3]  # Z_R=1: profiles (1,0) + (1,1)
    marg_R = np.column_stack([marg_R_0, marg_R_1])  # (N, 2)

    # Z_F marginal
    marg_F_0 = gamma[:, 0] + gamma[:, 2]  # Z_F=0: profiles (0,0) + (1,0)
    marg_F_1 = gamma[:, 1] + gamma[:, 3]  # Z_F=1: profiles (0,1) + (1,1)
    marg_F = np.column_stack([marg_F_0, marg_F_1])  # (N, 2)

    assert np.allclose(marg_R.sum(axis=1), 1.0, atol=1e-10), \
        "Z_R marginal rows must sum to 1"
    assert np.allclose(marg_F.sum(axis=1), 1.0, atol=1e-10), \
        "Z_F marginal rows must sum to 1"

    return marg_R, marg_F


def compute_product_marginals(
    gamma: np.ndarray,
) -> np.ndarray:
    """
    Compute the product-of-marginals approximation of the joint posterior.

    hat_q_i^{PM}(z_R, z_F) = hat_p_R,i^X(z_R) * hat_p_F,i^X(z_F)

    The approximation is over the 4 ordered profiles:
      q_i^{PM}[0] = P(Z_R=0) * P(Z_F=0)  [profile (0,0)]
      q_i^{PM}[1] = P(Z_R=0) * P(Z_F=1)  [profile (0,1)]
      q_i^{PM}[2] = P(Z_R=1) * P(Z_F=0)  [profile (1,0)]
      q_i^{PM}[3] = P(Z_R=1) * P(Z_F=1)  [profile (1,1)]

    Parameters
    ----------
    gamma : (N, 4) joint posterior; columns must correspond to ordered profiles

    Returns
    -------
    q_pm : (N, 4) product-of-marginals approximation; rows sum to 1.
    """
    marg_R, marg_F = compute_posterior_marginals(gamma)

    q_pm = np.column_stack([
        marg_R[:, 0] * marg_F[:, 0],  # (0,0)
        marg_R[:, 0] * marg_F[:, 1],  # (0,1)
        marg_R[:, 1] * marg_F[:, 0],  # (1,0)
        marg_R[:, 1] * marg_F[:, 1],  # (1,1)
    ])

    assert np.allclose(q_pm.sum(axis=1), 1.0, atol=1e-10), \
        "Product-of-marginals rows must sum to 1"
    return q_pm


# ---------------------------------------------------------------------------
# KL divergence: joint vs product-of-marginals
# ---------------------------------------------------------------------------

def kl_joint_vs_product(
    gamma: np.ndarray,
    q_pm: np.ndarray,
) -> np.ndarray:
    """
    Compute borrower-level KL divergence D_KL(hat_p_i^X || hat_q_i^{PM}).

    This quantity equals the posterior mutual information between Z_R and Z_F
    under the estimated joint posterior.

    Convention: 0 * log(0 / q) = 0.

    Parameters
    ----------
    gamma : (N, 4) joint posterior (hat_p_i^X)
    q_pm  : (N, 4) product-of-marginals approximation (hat_q_i^{PM})

    Returns
    -------
    kl : (N,) borrower-level KL divergences (non-negative)
    """
    # Clip q_pm for log — do not modify gamma (convention: 0*log(0/q)=0 via masking)
    q_safe = np.clip(q_pm, 1e-15, 1.0)
    g_safe = np.where(gamma > 0, gamma, 1.0)  # will be masked
    log_ratio = np.log(g_safe) - np.log(q_safe)
    kl = np.where(gamma > 0, gamma * log_ratio, 0.0).sum(axis=1)
    kl = np.maximum(kl, 0.0)  # numerical safeguard
    return kl


# ---------------------------------------------------------------------------
# PD from joint and product-of-marginals
# ---------------------------------------------------------------------------

def pd_from_joint(
    gamma: np.ndarray,
    p_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute borrower-level PD from the joint posterior.

    hat_PD_i = sum_k hat_p_k hat_gamma_ik^X

    Parameters
    ----------
    gamma : (N, 4) joint posterior
    p_vec : (4,)   profile-specific default probabilities

    Returns
    -------
    pd : (N,) borrower-level PDs
    """
    return gamma @ p_vec  # (N,)


def pd_from_product(
    q_pm: np.ndarray,
    p_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute borrower-level PD from the product-of-marginals approximation.

    hat_PD_i^{PM} = sum_{z_R,z_F} hat_q_i^{PM}(z_R,z_F) * hat_p_{z_R,z_F}

    where hat_p_{0,0}=p_1, hat_p_{0,1}=p_2, hat_p_{1,0}=p_3, hat_p_{1,1}=p_4.

    Parameters
    ----------
    q_pm  : (N, 4) product-of-marginals approximation
    p_vec : (4,)   profile-specific default probabilities (aligned)

    Returns
    -------
    pd_pm : (N,) borrower-level PDs from factorized approximation
    """
    return q_pm @ p_vec  # (N,)


# ---------------------------------------------------------------------------
# Summary metrics container and computation
# ---------------------------------------------------------------------------

@dataclass
class ProductMarginalsMetrics:
    """Metrics for product-of-marginals approximation analysis."""
    kl_avg: float       # average borrower-level KL(joint || product-marginals)
    mae_pd_pm: float    # average absolute PD distortion from factorization


def summarize_pm_metrics(
    gamma: np.ndarray,
    p_vec: np.ndarray,
) -> tuple[ProductMarginalsMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all product-of-marginals metrics for the experiment.

    Parameters
    ----------
    gamma : (N, 4) aligned estimated score-based joint posterior hat_gamma_i^X
    p_vec : (4,)   aligned estimated profile-specific default probabilities

    Returns
    -------
    metrics    : ProductMarginalsMetrics with headline metrics
    q_pm       : (N, 4) product-of-marginals approximation
    kl_values  : (N,)   per-borrower KL divergences
    pd_joint   : (N,)   PDs from joint posterior
    pd_pm      : (N,)   PDs from factorized approximation
    """
    # Validate input
    assert gamma.shape[1] == 4, "gamma must have 4 columns"
    assert p_vec.shape[0] == 4, "p_vec must have 4 elements"
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-9), \
        "Posterior rows must sum to 1"

    q_pm = compute_product_marginals(gamma)
    kl_values = kl_joint_vs_product(gamma, q_pm)

    pd_joint = pd_from_joint(gamma, p_vec)
    pd_pm = pd_from_product(q_pm, p_vec)

    avg_kl = float(kl_values.mean())
    avg_mae_pd = float(np.mean(np.abs(pd_pm - pd_joint)))

    metrics = ProductMarginalsMetrics(
        kl_avg=avg_kl,
        mae_pd_pm=avg_mae_pd,
    )

    return metrics, q_pm, kl_values, pd_joint, pd_pm
