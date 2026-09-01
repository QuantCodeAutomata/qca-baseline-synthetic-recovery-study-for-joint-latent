"""
Joint latent-state EM estimator for Gaussian mixture scores + Bernoulli defaults.

Maximizes the observed-data likelihood:
  L(Theta; X, D) = prod_i sum_k pi_k phi_2(X_i; mu_k, Sigma_k)
                            * p_k^{D_i} * (1-p_k)^{1-D_i}

Algorithm:
  - 20 multi-start configurations
  - KMeans initialization (K=4, 5 internal starts)
  - 3% label perturbation
  - EM up to 300 iterations per run
  - Convergence: |ell_{t+1} - ell_t| / (1 + |ell_t|) < 1e-8
  - Discard runs with ESS < 10 for any component
  - Select best run by highest observed-data log-likelihood

Custom — Context7 found no library equivalent for this exact joint EM with
Bernoulli defaults and Gaussian score mixture (paper Eq. described in methodology).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import logsumexp  # Context7 confirmed
from sklearn.cluster import KMeans   # Context7 confirmed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K: int = 4             # number of latent profiles
D: int = 2             # score dimension
N_RUNS: int = 20       # number of multi-start configurations
MAX_ITER: int = 300    # maximum EM iterations per run
CONV_TOL: float = 1e-8 # convergence tolerance
MIN_ESS: float = 10.0  # minimum effective sample size per component
COV_RIDGE: float = 1e-6  # covariance regularization
P_EPS: float = 1e-6    # clipping bounds for p_k
KMEANS_INIT: int = 5   # KMeans internal restarts
PERTURB_FRAC: float = 0.03  # fraction of labels to perturb randomly


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EMRunResult:
    """Result from a single EM run."""
    converged: bool
    n_iter: int
    log_likelihood: float
    pi: np.ndarray          # (K,)
    mu: np.ndarray          # (K, D)
    sigma: list[np.ndarray] # K x (D, D)
    p: np.ndarray           # (K,)
    tau: np.ndarray         # (N, K) EM responsibilities
    ll_history: list[float] = field(default_factory=list)
    min_ess: float = 0.0
    admissible: bool = False


@dataclass
class EMEstimatorResult:
    """Final result after multi-start EM + best-run selection."""
    pi: np.ndarray          # (K,)   estimated mixture weights
    mu: np.ndarray          # (K, D) estimated component means
    sigma: list[np.ndarray] # K x (D, D) estimated covariances
    p: np.ndarray           # (K,)   estimated default probabilities
    tau: np.ndarray         # (N, K) EM responsibilities from best run
    log_likelihood: float   # observed-data log-likelihood of best run
    n_admissible_runs: int
    best_run_index: int
    all_runs: list[EMRunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _mvn_logpdf_batch(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Numerically stable multivariate normal log-pdf using Cholesky decomposition.

    Computes log phi_2(X_i; mu, Sigma) for all i simultaneously.

    Parameters
    ----------
    X     : (N, D) observation matrix
    mu    : (D,)   mean vector
    sigma : (D, D) covariance matrix

    Returns
    -------
    log_probs : (N,) log-probability values
    """
    D = mu.shape[0]
    diff = X - mu  # (N, D)
    try:
        L = np.linalg.cholesky(sigma)  # (D, D) lower triangular
    except np.linalg.LinAlgError:
        # Fallback: add ridge and retry
        sigma_r = sigma + COV_RIDGE * np.eye(D)
        L = np.linalg.cholesky(sigma_r)

    # Solve L @ alpha = diff.T  =>  alpha = L^{-1} diff.T
    alpha = np.linalg.solve(L, diff.T)  # (D, N)
    maha = np.sum(alpha ** 2, axis=0)   # (N,) Mahalanobis distances squared

    log_det = 2.0 * np.sum(np.log(np.diag(L)))  # log |Sigma|
    log_const = -0.5 * (D * np.log(2 * np.pi) + log_det)
    return log_const - 0.5 * maha


def _compute_log_responsibilities(
    X: np.ndarray,
    D_obs: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: list[np.ndarray],
    p: np.ndarray,
) -> np.ndarray:
    """
    E-step: compute log unnormalized responsibilities.

    log_tau_unnorm[i, k] = log pi_k + log phi_2(X_i; mu_k, Sigma_k)
                          + D_i log p_k + (1-D_i) log(1-p_k)

    Parameters
    ----------
    X     : (N, 2) score matrix
    D_obs : (N,)   default indicators
    pi    : (K,)   mixture weights
    mu    : (K, D) component means
    sigma : list of K covariance matrices
    p     : (K,)   default probabilities

    Returns
    -------
    log_tau : (N, K) unnormalized log-responsibilities
    """
    N = X.shape[0]
    K = pi.shape[0]
    log_tau = np.zeros((N, K), dtype=float)

    log_p = np.log(np.clip(p, P_EPS, 1 - P_EPS))
    log_1mp = np.log(np.clip(1 - p, P_EPS, 1 - P_EPS))

    for k in range(K):
        log_mix = np.log(pi[k])
        log_phi = _mvn_logpdf_batch(X, mu[k], sigma[k])
        # Bernoulli log-likelihood: D_i * log(p_k) + (1-D_i) * log(1-p_k)
        log_bern = D_obs * log_p[k] + (1 - D_obs) * log_1mp[k]
        log_tau[:, k] = log_mix + log_phi + log_bern

    return log_tau


def _normalize_log_responsibilities(log_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize log-responsibilities using log-sum-exp.

    # Using scipy.special.logsumexp — Context7 confirmed

    Parameters
    ----------
    log_tau : (N, K) unnormalized log-responsibilities

    Returns
    -------
    tau        : (N, K) normalized responsibilities (sum to 1 per row)
    log_sum    : (N,)   log-normalizer for each borrower
    """
    log_sum = logsumexp(log_tau, axis=1, keepdims=True)  # (N, 1)
    tau = np.exp(log_tau - log_sum)                       # (N, K)
    return tau, log_sum.squeeze()


def _observed_data_loglik(log_tau_unnorm: np.ndarray) -> float:
    """
    Compute observed-data log-likelihood from unnormalized log-responsibilities.

    ell = sum_i log sum_k exp(log_tau_unnorm[i, k])
    """
    return float(logsumexp(log_tau_unnorm, axis=1).sum())


# ---------------------------------------------------------------------------
# M-step update
# ---------------------------------------------------------------------------

def _m_step(
    X: np.ndarray,
    D_obs: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    M-step: update parameters given responsibilities tau.

    N_k  = sum_i tau_ik
    pi_k = N_k / n
    mu_k = sum_i tau_ik X_i / N_k
    Sigma_k = [sum_i tau_ik (X_i - mu_k)(X_i - mu_k)^T] / N_k  (+ ridge)
    p_k  = sum_i tau_ik D_i / N_k  (clipped to [P_EPS, 1-P_EPS])

    Parameters
    ----------
    X     : (N, 2) score matrix
    D_obs : (N,)   default indicators
    tau   : (N, K) responsibilities

    Returns
    -------
    pi    : (K,)
    mu    : (K, D)
    sigma : list of K (D, D) matrices
    p     : (K,)
    """
    N, d = X.shape
    K = tau.shape[1]
    Nk = tau.sum(axis=0)  # (K,)

    pi = Nk / N

    mu = np.zeros((K, d), dtype=float)
    sigma = []
    p = np.zeros(K, dtype=float)

    for k in range(K):
        w = tau[:, k]          # (N,)
        mu[k] = (w[:, None] * X).sum(axis=0) / Nk[k]

        diff = X - mu[k]       # (N, d)
        cov_k = (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / Nk[k]
        cov_k += COV_RIDGE * np.eye(d)  # regularize for positive definiteness
        sigma.append(cov_k)

        p[k] = np.dot(w, D_obs) / Nk[k]

    # Clip default probabilities
    p = np.clip(p, P_EPS, 1 - P_EPS)

    return pi, mu, sigma, p


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _initialize_from_kmeans(
    X: np.ndarray,
    D_obs: np.ndarray,
    run_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Build initial parameters from KMeans clustering with label perturbation.

    Steps:
    1. Run KMeans(n_clusters=K, n_init=5) on X.
    2. Perturb PERTURB_FRAC of labels randomly.
    3. Compute initial pi, mu, Sigma, p from perturbed partition.

    # Using sklearn.cluster.KMeans — Context7 confirmed

    Parameters
    ----------
    X        : (N, 2) score matrix
    D_obs    : (N,)   default indicators
    run_seed : random seed for this specific run

    Returns
    -------
    pi, mu, sigma, p — initial parameter estimates
    """
    rng = np.random.default_rng(run_seed)

    km = KMeans(n_clusters=K, n_init=KMEANS_INIT, random_state=run_seed)
    labels = km.fit_predict(X)  # (N,) cluster assignments in {0,...,K-1}

    # Perturb 3% of labels randomly
    N = X.shape[0]
    n_perturb = max(1, int(PERTURB_FRAC * N))
    perturb_idx = rng.choice(N, size=n_perturb, replace=False)
    labels[perturb_idx] = rng.integers(0, K, size=n_perturb)

    # Build initial parameters from perturbed partition
    d = X.shape[1]
    pi = np.zeros(K, dtype=float)
    mu = np.zeros((K, d), dtype=float)
    sigma = []
    p = np.zeros(K, dtype=float)

    for k in range(K):
        mask = labels == k
        nk = mask.sum()
        if nk < 2:
            # Degenerate cluster: fall back to identity
            pi[k] = 1.0 / K
            mu[k] = X.mean(axis=0)
            sigma.append(np.eye(d) * 0.5 + COV_RIDGE * np.eye(d))
            p[k] = D_obs.mean()
        else:
            pi[k] = nk / N
            mu[k] = X[mask].mean(axis=0)
            cov_k = np.cov(X[mask].T, bias=True) + COV_RIDGE * np.eye(d)
            sigma.append(cov_k)
            p[k] = D_obs[mask].mean()

    # Ensure pi sums to 1 (may drift slightly due to rounding)
    pi /= pi.sum()
    p = np.clip(p, P_EPS, 1 - P_EPS)

    return pi, mu, sigma, p


# ---------------------------------------------------------------------------
# Single EM run
# ---------------------------------------------------------------------------

def _run_em(
    X: np.ndarray,
    D_obs: np.ndarray,
    pi_init: np.ndarray,
    mu_init: np.ndarray,
    sigma_init: list[np.ndarray],
    p_init: np.ndarray,
) -> EMRunResult:
    """
    Run EM algorithm from a single initialization until convergence or max iterations.

    Parameters
    ----------
    X          : (N, 2) score matrix
    D_obs      : (N,)   default indicators
    pi_init    : (K,)   initial mixture weights
    mu_init    : (K, D) initial means
    sigma_init : K x (D, D) initial covariances
    p_init     : (K,)   initial default probabilities

    Returns
    -------
    EMRunResult with all relevant quantities.
    """
    pi = pi_init.copy()
    mu = mu_init.copy()
    sigma = [s.copy() for s in sigma_init]
    p = p_init.copy()

    ll_history: list[float] = []
    converged = False
    n_iter = 0
    ll_prev = -np.inf

    for t in range(MAX_ITER):
        # E-step
        log_tau = _compute_log_responsibilities(X, D_obs, pi, mu, sigma, p)
        ll_curr = _observed_data_loglik(log_tau)
        ll_history.append(ll_curr)

        # Convergence check
        rel_change = abs(ll_curr - ll_prev) / (1.0 + abs(ll_prev))
        if t > 0 and rel_change < CONV_TOL:
            converged = True
            n_iter = t + 1
            break

        ll_prev = ll_curr

        # Normalize responsibilities
        tau, _ = _normalize_log_responsibilities(log_tau)

        # M-step
        pi, mu, sigma, p = _m_step(X, D_obs, tau)
        n_iter = t + 1

    # Final E-step to get tau at converged parameters
    log_tau = _compute_log_responsibilities(X, D_obs, pi, mu, sigma, p)
    ll_final = _observed_data_loglik(log_tau)
    tau, _ = _normalize_log_responsibilities(log_tau)

    # Effective sample size per component
    Nk = tau.sum(axis=0)
    min_ess = float(Nk.min())

    # Admissibility check: ESS >= MIN_ESS for all components and converged
    admissible = converged and (min_ess >= MIN_ESS)

    return EMRunResult(
        converged=converged,
        n_iter=n_iter,
        log_likelihood=ll_final,
        pi=pi,
        mu=mu,
        sigma=sigma,
        p=p,
        tau=tau,
        ll_history=ll_history,
        min_ess=min_ess,
        admissible=admissible,
    )


# ---------------------------------------------------------------------------
# Multi-start EM
# ---------------------------------------------------------------------------

def fit_joint_em(
    X: np.ndarray,
    D_obs: np.ndarray,
    base_seed: int = 42,
    verbose: bool = False,
) -> EMEstimatorResult:
    """
    Run the full multi-start joint EM procedure and return the best admissible run.

    Multi-start strategy (exact methodology):
    1. For each of N_RUNS=20 configurations:
       a. Build initialization via KMeans + label perturbation.
       b. Run EM up to MAX_ITER=300 iterations.
    2. Discard runs that are not admissible (not converged or ESS < 10).
    3. Among admissible runs, select the one with highest observed-data log-likelihood.

    Parameters
    ----------
    X         : (N, 2) score matrix (observed scores only)
    D_obs     : (N,)   default indicators (observed)
    base_seed : base random seed for initialization diversity
    verbose   : print run diagnostics if True

    Returns
    -------
    EMEstimatorResult with best-run parameters.

    Raises
    ------
    RuntimeError if no admissible run found.
    """
    all_runs: list[EMRunResult] = []

    for run_idx in range(N_RUNS):
        run_seed = base_seed + run_idx * 997  # distinct seeds per run

        # Build initialization
        pi_init, mu_init, sigma_init, p_init = _initialize_from_kmeans(
            X, D_obs, run_seed
        )

        # Run EM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _run_em(X, D_obs, pi_init, mu_init, sigma_init, p_init)

        all_runs.append(result)

        if verbose:
            print(
                f"Run {run_idx+1:2d}/{N_RUNS}: converged={result.converged}, "
                f"n_iter={result.n_iter:3d}, min_ESS={result.min_ess:.1f}, "
                f"ll={result.log_likelihood:.4f}, admissible={result.admissible}"
            )

    # Filter admissible runs
    admissible_runs = [r for r in all_runs if r.admissible]
    n_admissible = len(admissible_runs)

    if n_admissible == 0:
        raise RuntimeError(
            f"No admissible EM runs found (converged + ESS >= {MIN_ESS}). "
            "Check initialization or relax convergence criteria."
        )

    # Select best run by highest log-likelihood
    best_run = max(admissible_runs, key=lambda r: r.log_likelihood)
    best_idx = all_runs.index(best_run)

    if verbose:
        print(
            f"\nBest run: index={best_idx}, ll={best_run.log_likelihood:.6f}, "
            f"n_admissible={n_admissible}/{N_RUNS}"
        )

    return EMEstimatorResult(
        pi=best_run.pi,
        mu=best_run.mu,
        sigma=best_run.sigma,
        p=best_run.p,
        tau=best_run.tau,
        log_likelihood=best_run.log_likelihood,
        n_admissible_runs=n_admissible,
        best_run_index=best_idx,
        all_runs=all_runs,
    )
