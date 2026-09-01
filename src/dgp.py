"""
Data-generating process (DGP) for the joint latent-state EM credit risk study.

Generates N borrower-level observations from a 4-component Gaussian mixture
with profile-specific Bernoulli default probabilities.

Profile ordering (fixed throughout):
  k=0 -> (Z_R=0, Z_F=0)  [1-indexed: profile 1]
  k=1 -> (Z_R=0, Z_F=1)  [1-indexed: profile 2]
  k=2 -> (Z_R=1, Z_F=0)  [1-indexed: profile 3]
  k=3 -> (Z_R=1, Z_F=1)  [1-indexed: profile 4]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# True parameter set Theta* (exact values from paper)
# ---------------------------------------------------------------------------

TRUE_PI: np.ndarray = np.array([0.40, 0.20, 0.25, 0.15])

TRUE_MU: np.ndarray = np.array([
    [1.8,  0.3],   # profile 1: (Z_R=0, Z_F=0)
    [1.8,  1.8],   # profile 2: (Z_R=0, Z_F=1)
    [-0.8, 0.3],   # profile 3: (Z_R=1, Z_F=0)
    [-0.8, 1.8],   # profile 4: (Z_R=1, Z_F=1)
], dtype=float)

TRUE_SIGMA: list[np.ndarray] = [
    np.array([[0.40, 0.10], [0.10, 0.40]]),  # profile 1
    np.array([[0.40, 0.10], [0.10, 0.40]]),  # profile 2
    np.array([[0.45, 0.10], [0.10, 0.45]]),  # profile 3
    np.array([[0.45, 0.10], [0.10, 0.45]]),  # profile 4
]

TRUE_P: np.ndarray = np.array([0.01, 0.05, 0.10, 0.25])

# Latent binary dimensions for each profile (Z_R, Z_F)
PROFILE_DIMS: np.ndarray = np.array([
    [0, 0],  # profile 1
    [0, 1],  # profile 2
    [1, 0],  # profile 3
    [1, 1],  # profile 4
], dtype=int)


@dataclass
class TrueParameters:
    """Container for true DGP parameters."""
    pi: np.ndarray = field(default_factory=lambda: TRUE_PI.copy())
    mu: np.ndarray = field(default_factory=lambda: TRUE_MU.copy())
    sigma: list[np.ndarray] = field(
        default_factory=lambda: [s.copy() for s in TRUE_SIGMA]
    )
    p: np.ndarray = field(default_factory=lambda: TRUE_P.copy())
    K: int = 4
    D: int = 2  # score dimension


@dataclass
class SimulatedData:
    """Container for all simulated borrower data (observed + latent for validation)."""
    # Observed variables (estimation input)
    X: np.ndarray        # (N, 2) score vectors [R_i, F_i]
    D: np.ndarray        # (N,)   default indicators

    # Latent variables (hidden from estimation, kept for validation only)
    Z: np.ndarray        # (N,)   latent profile indices in {0,1,2,3}
    Z_R: np.ndarray      # (N,)   latent creditworthiness state
    Z_F: np.ndarray      # (N,)   latent fragility state

    # True posteriors and PDs (computed from true parameters)
    gamma_X_true: np.ndarray  # (N, 4) score-based posteriors gamma_ik^{X,*}
    PD_true: np.ndarray       # (N,)   true borrower-level PDs PD_i^*

    # Reference to true parameters
    theta_star: TrueParameters = field(default_factory=TrueParameters)
    N: int = 5000
    seed: int = 1234


def mvn_logpdf(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute multivariate normal log-pdf for each row of X.

    # Using scipy.stats.multivariate_normal.logpdf — Context7 confirmed

    Parameters
    ----------
    X     : (N, D) array of observation vectors
    mu    : (D,)   mean vector
    sigma : (D, D) covariance matrix

    Returns
    -------
    log_probs : (N,) array of log-probabilities
    """
    return np.atleast_1d(
        multivariate_normal.logpdf(X, mean=mu, cov=sigma, allow_singular=False)
    )


def compute_true_score_posteriors(
    X: np.ndarray,
    theta: TrueParameters,
) -> np.ndarray:
    """
    Compute true score-based posterior beliefs gamma_i^{X,*}.

    gamma_ik^{X,*} = pi_k* phi_2(X_i; mu_k*, Sigma_k*) /
                     sum_j pi_j* phi_2(X_j; mu_j*, Sigma_j*)

    Parameters
    ----------
    X     : (N, 2) score matrix
    theta : TrueParameters

    Returns
    -------
    gamma : (N, K) posterior matrix; rows sum to 1.
    """
    K = theta.K
    N = X.shape[0]

    # Log-unnormalized responsibilities: log pi_k + log phi_2(X_i; mu_k, Sigma_k)
    log_resp = np.zeros((N, K), dtype=float)
    for k in range(K):
        log_resp[:, k] = np.log(theta.pi[k]) + mvn_logpdf(X, theta.mu[k], theta.sigma[k])

    # Normalize using log-sum-exp for numerical stability
    # Custom — Context7 found no library equivalent for this exact broadcasting pattern
    log_normalizer = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
    gamma = np.exp(log_resp - log_normalizer)

    # Verify rows sum to 1
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10), \
        "Score-based posterior rows must sum to 1"
    return gamma


def simulate(N: int = 5000, seed: int = 1234) -> SimulatedData:
    """
    Generate N iid borrowers from the true DGP defined in the paper.

    Steps (exact methodology):
    1. Fix random seed.
    2. For each borrower i:
       a. Draw latent profile Z_i ~ Categorical(pi*).
       b. Map to latent dimensions (Z_R_i, Z_F_i).
       c. Draw X_i | Z_i=k ~ N_2(mu_k*, Sigma_k*).
       d. Draw D_i | Z_i=k ~ Bernoulli(p_k*).
    3. Compute true score-based posteriors gamma_i^{X,*} using true parameters.
    4. Compute true borrower-level PD_i^* = sum_k p_k* gamma_ik^{X,*}.

    Parameters
    ----------
    N    : number of borrowers (default 5000)
    seed : random seed (default 1234)

    Returns
    -------
    SimulatedData dataclass with all fields populated.
    """
    rng = np.random.default_rng(seed)
    theta = TrueParameters()
    K = theta.K

    # Step 1: Draw latent profile indices Z_i ~ Categorical(pi*)
    Z = rng.choice(K, size=N, p=theta.pi)  # 0-indexed

    # Step 2: Map to latent binary dimensions
    Z_R = PROFILE_DIMS[Z, 0]
    Z_F = PROFILE_DIMS[Z, 1]

    # Step 3: Draw score vectors X_i | Z_i=k ~ N_2(mu_k*, Sigma_k*)
    X = np.zeros((N, 2), dtype=float)
    for k in range(K):
        mask = Z == k
        n_k = mask.sum()
        if n_k > 0:
            X[mask] = rng.multivariate_normal(theta.mu[k], theta.sigma[k], size=n_k)

    # Step 4: Draw default indicators D_i | Z_i=k ~ Bernoulli(p_k*)
    D = np.zeros(N, dtype=int)
    for k in range(K):
        mask = Z == k
        n_k = mask.sum()
        if n_k > 0:
            D[mask] = rng.binomial(1, theta.p[k], size=n_k)

    # Step 5: Compute true score-based posterior beliefs gamma_i^{X,*}
    gamma_X_true = compute_true_score_posteriors(X, theta)

    # Step 6: Compute true borrower-level PDs
    PD_true = gamma_X_true @ theta.p  # (N,)

    return SimulatedData(
        X=X,
        D=D,
        Z=Z,
        Z_R=Z_R,
        Z_F=Z_F,
        gamma_X_true=gamma_X_true,
        PD_true=PD_true,
        theta_star=theta,
        N=N,
        seed=seed,
    )
