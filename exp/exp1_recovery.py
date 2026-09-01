"""
Experiment 1: Baseline synthetic recovery study for joint latent-state EM
              in credit risk.

Reproduces the paper's controlled simulation experiment:
  - Generates N=5000 borrowers from a 4-component Gaussian mixture with
    profile-specific Bernoulli default probabilities.
  - Fits a joint EM algorithm to (X_i, D_i) observed pairs.
  - Aligns estimated labels to true labels.
  - Computes score-based predictive posteriors and borrower-level PDs.
  - Evaluates all ten recovery metrics.
  - Saves tables, parameter summaries, and plots to results/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dgp import simulate, TrueParameters, TRUE_PI, TRUE_MU, TRUE_SIGMA, TRUE_P
from src.em_estimator import fit_joint_em
from src.label_alignment import align_estimates
from src.posteriors import compute_score_posteriors, compute_borrower_pd
from src.metrics import compute_all_metrics, RecoveryMetrics

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_exp1(verbose: bool = True) -> dict:
    """
    Execute the complete Experiment 1 pipeline.

    Returns
    -------
    dict with all key results for downstream use (e.g., Experiment 2).
    """
    print("=" * 70)
    print("Experiment 1: Joint Latent-State EM Recovery Study")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data (seed=1234, N=5000)
    # ------------------------------------------------------------------
    print("\n[1/6] Simulating N=5000 borrowers (seed=1234)...")
    data = simulate(N=5000, seed=1234)
    print(f"      Default rate: {data.D.mean():.4f}")
    print(f"      True PD range: [{data.PD_true.min():.4f}, {data.PD_true.max():.4f}]")

    # Empirical profile counts
    for k in range(4):
        n_k = (data.Z == k).sum()
        print(f"      Profile {k+1}: n={n_k} ({n_k/data.N*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 2: Fit joint EM to observed (X, D) pairs only
    # ------------------------------------------------------------------
    print("\n[2/6] Fitting joint EM (20 restarts, max 300 iter each)...")
    em_result = fit_joint_em(data.X, data.D, base_seed=42, verbose=verbose)
    print(f"      Admissible runs: {em_result.n_admissible_runs}/20")
    print(f"      Best run log-likelihood: {em_result.log_likelihood:.6f}")
    print(f"      Best run index: {em_result.best_run_index}")

    # ------------------------------------------------------------------
    # Step 3: Align estimated labels to true labels (Hungarian algorithm)
    # ------------------------------------------------------------------
    print("\n[3/6] Aligning estimated labels to true labels...")
    theta_star = data.theta_star
    pi_hat, mu_hat, sigma_hat, p_hat, tau_hat = align_estimates(
        em_result.pi,
        em_result.mu,
        em_result.sigma,
        em_result.p,
        em_result.tau,
        theta_star.mu,
    )

    print("      Aligned estimated pi:  ", np.round(pi_hat, 4))
    print("      True              pi:  ", np.round(theta_star.pi, 4))
    print("      Aligned estimated p:   ", np.round(p_hat, 4))
    print("      True              p:   ", np.round(theta_star.p, 4))

    # ------------------------------------------------------------------
    # Step 4: Compute score-based predictive posteriors (D_i not used)
    # ------------------------------------------------------------------
    print("\n[4/6] Computing score-based predictive posteriors...")
    gamma_hat = compute_score_posteriors(data.X, pi_hat, mu_hat, sigma_hat)
    pd_hat = compute_borrower_pd(gamma_hat, p_hat)

    print(f"      Estimated PD range: [{pd_hat.min():.4f}, {pd_hat.max():.4f}]")

    # ------------------------------------------------------------------
    # Step 5: Compute all recovery metrics
    # ------------------------------------------------------------------
    print("\n[5/6] Computing recovery metrics...")
    metrics = compute_all_metrics(
        pi_hat=pi_hat,
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        p_hat=p_hat,
        gamma_hat=gamma_hat,
        pd_hat=pd_hat,
        pi_true=theta_star.pi,
        mu_true=theta_star.mu,
        sigma_true=theta_star.sigma,
        p_true=theta_star.p,
        gamma_true=data.gamma_X_true,
        pd_true=data.PD_true,
    )

    _print_metrics(metrics)

    # ------------------------------------------------------------------
    # Step 6: Save results (tables + plots)
    # ------------------------------------------------------------------
    print("\n[6/6] Saving results to results/...")
    _save_parameter_table(theta_star, pi_hat, mu_hat, sigma_hat, p_hat)
    _save_metrics_table(metrics)
    _save_plots(data, gamma_hat, pd_hat, metrics, em_result)

    results = {
        "metrics": metrics,
        "pi_hat": pi_hat,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "p_hat": p_hat,
        "tau_hat": tau_hat,
        "gamma_hat": gamma_hat,
        "pd_hat": pd_hat,
        "data": data,
        "em_result": em_result,
    }
    print("\nExperiment 1 complete.")
    return results


# ---------------------------------------------------------------------------
# Helper: print metrics
# ---------------------------------------------------------------------------

def _print_metrics(metrics: RecoveryMetrics) -> None:
    """Print recovery metrics in a readable format."""
    print("\n  Recovery Metrics:")
    print(f"    MAE_pi    = {metrics.MAE_pi:.4e}  (target ~4.591e-3)")
    print(f"    MSE_mu    = {metrics.MSE_mu:.4e}  (target ~1.170e-3)")
    print(f"    MSE_Sigma = {metrics.MSE_Sigma:.4e}  (target ~2.157e-3)")
    print(f"    MAE_p     = {metrics.MAE_p:.4e}  (target ~3.263e-3)")
    print(f"    MSE_p     = {metrics.MSE_p:.4e}  (target ~1.902e-5)")
    print(f"    MSE_gamma = {metrics.MSE_gamma:.4e}  (target ~8.163e-4)")
    print(f"    KL_avg    = {metrics.KL_avg:.4e}  (target ~1.736e-3)")
    print(f"    JS_avg    = {metrics.JS_avg:.4e}  (target ~4.367e-4)")
    print(f"    MSE_PD    = {metrics.MSE_PD:.4e}  (target ~1.419e-5)")
    print(f"    MAE_PD    = {metrics.MAE_PD:.4e}  (target ~2.691e-3)")


# ---------------------------------------------------------------------------
# Helper: save parameter comparison table
# ---------------------------------------------------------------------------

def _save_parameter_table(
    theta_star: TrueParameters,
    pi_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: list[np.ndarray],
    p_hat: np.ndarray,
) -> None:
    """Save parameter recovery comparison table as CSV."""
    K = theta_star.K
    rows = []
    for k in range(K):
        rows.append({
            "Profile": k + 1,
            "pi_true": theta_star.pi[k],
            "pi_hat": pi_hat[k],
            "mu_R_true": theta_star.mu[k, 0],
            "mu_R_hat": mu_hat[k, 0],
            "mu_F_true": theta_star.mu[k, 1],
            "mu_F_hat": mu_hat[k, 1],
            "p_true": theta_star.p[k],
            "p_hat": p_hat[k],
            "Sigma11_true": theta_star.sigma[k][0, 0],
            "Sigma11_hat": sigma_hat[k][0, 0],
            "Sigma12_true": theta_star.sigma[k][0, 1],
            "Sigma12_hat": sigma_hat[k][0, 1],
            "Sigma22_true": theta_star.sigma[k][1, 1],
            "Sigma22_hat": sigma_hat[k][1, 1],
        })
    df = pd.DataFrame(rows).round(6)
    df.to_csv(RESULTS_DIR / "exp1_parameter_recovery.csv", index=False)
    print(f"    Saved: {RESULTS_DIR / 'exp1_parameter_recovery.csv'}")


# ---------------------------------------------------------------------------
# Helper: save metrics table
# ---------------------------------------------------------------------------

def _save_metrics_table(metrics: RecoveryMetrics) -> None:
    """Save recovery metrics to CSV."""
    target_vals = {
        "MAE_pi": 4.591e-3,
        "MSE_mu": 1.170e-3,
        "MSE_Sigma": 2.157e-3,
        "MAE_p": 3.263e-3,
        "MSE_p": 1.902e-5,
        "MSE_gamma": 8.163e-4,
        "KL_avg": 1.736e-3,
        "JS_avg": 4.367e-4,
        "MSE_PD": 1.419e-5,
        "MAE_PD": 2.691e-3,
    }
    achieved = {
        "MAE_pi": metrics.MAE_pi,
        "MSE_mu": metrics.MSE_mu,
        "MSE_Sigma": metrics.MSE_Sigma,
        "MAE_p": metrics.MAE_p,
        "MSE_p": metrics.MSE_p,
        "MSE_gamma": metrics.MSE_gamma,
        "KL_avg": metrics.KL_avg,
        "JS_avg": metrics.JS_avg,
        "MSE_PD": metrics.MSE_PD,
        "MAE_PD": metrics.MAE_PD,
    }
    rows = [
        {
            "Metric": name,
            "Achieved": f"{achieved[name]:.4e}",
            "Target (paper)": f"{target_vals[name]:.4e}",
        }
        for name in target_vals
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp1_recovery_metrics.csv", index=False)
    print(f"    Saved: {RESULTS_DIR / 'exp1_recovery_metrics.csv'}")


# ---------------------------------------------------------------------------
# Helper: save diagnostic plots
# ---------------------------------------------------------------------------

def _save_plots(
    data,
    gamma_hat: np.ndarray,
    pd_hat: np.ndarray,
    metrics: RecoveryMetrics,
    em_result=None,
) -> None:
    """Save diagnostic plots for Experiment 1."""

    # Plot 1: Scatter of R vs F colored by true profile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = ["steelblue", "darkorange", "green", "red"]
    labels = ["Profile 1 (0,0)", "Profile 2 (0,1)", "Profile 3 (1,0)", "Profile 4 (1,1)"]
    for k in range(4):
        mask = data.Z == k
        ax.scatter(
            data.X[mask, 0], data.X[mask, 1],
            c=colors[k], alpha=0.3, s=10, label=labels[k]
        )
    ax.set_xlabel("R (creditworthiness score)")
    ax.set_ylabel("F (fragility score)")
    ax.set_title("Simulated Borrower Scores by True Profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: True PD vs Estimated PD scatter
    ax = axes[1]
    ax.scatter(data.PD_true, pd_hat, alpha=0.2, s=5, c="navy")
    lims = [min(data.PD_true.min(), pd_hat.min()), max(data.PD_true.max(), pd_hat.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect recovery")
    ax.set_xlabel("True PD_i*")
    ax.set_ylabel("Estimated hat_PD_i")
    ax.set_title(f"Borrower-Level PD Recovery\nMSE={metrics.MSE_PD:.4e}, MAE={metrics.MAE_PD:.4e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {RESULTS_DIR / 'exp1_overview.png'}")

    # Plot 3: Posterior recovery per profile
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for k in range(4):
        ax = axes[k]
        ax.scatter(
            data.gamma_X_true[:, k], gamma_hat[:, k],
            alpha=0.2, s=5, c=colors[k]
        )
        ax.plot([0, 1], [0, 1], "r--", lw=1.5)
        ax.set_xlabel(f"True gamma*_{{i{k+1}}}^X")
        ax.set_ylabel(f"Est. hat_gamma_{{i{k+1}}}^X")
        ax.set_title(f"Posterior Profile {k+1}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Score-Based Posterior Recovery (True vs Estimated)", fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_posterior_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {RESULTS_DIR / 'exp1_posterior_recovery.png'}")

    # Plot 4: EM log-likelihood convergence (best run)
    fig, ax = plt.subplots(figsize=(8, 4))
    if em_result is not None:
        best_run = em_result.all_runs[em_result.best_run_index]
        ax.plot(best_run.ll_history, "b-", lw=1.5)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Observed-Data Log-Likelihood")
    ax.set_title("EM Convergence (Best Run)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_em_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {RESULTS_DIR / 'exp1_em_convergence.png'}")


if __name__ == "__main__":
    results = run_exp1(verbose=True)
