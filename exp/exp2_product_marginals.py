"""
Experiment 2: Product-of-marginals approximation analysis for borrower-level
              latent posterior beliefs.

Analyzes information loss from approximating the joint posterior over latent
creditworthiness (Z_R) and fragility (Z_F) by the product of its marginals.

Requires results from Experiment 1 as input (gamma_hat, p_hat).

Key metrics:
  - Average KL divergence D_KL(joint || product-marginals)
  - Average absolute PD distortion from factorization (MAE_PD^{PM})
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.product_marginals import (
    summarize_pm_metrics,
    compute_posterior_marginals,
    compute_product_marginals,
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp2(
    gamma_hat: np.ndarray,
    p_hat: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Execute Experiment 2: product-of-marginals approximation analysis.

    Parameters
    ----------
    gamma_hat : (N, 4) aligned estimated score-based joint posterior
    p_hat     : (4,)   aligned estimated profile-specific default probabilities

    Returns
    -------
    dict with metrics and arrays for downstream use.
    """
    print("\n" + "=" * 70)
    print("Experiment 2: Product-of-Marginals Approximation Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Compute product-of-marginals approximation and metrics
    # ------------------------------------------------------------------
    print("\n[1/3] Computing product-of-marginals approximation...")
    metrics, q_pm, kl_values, pd_joint, pd_pm = summarize_pm_metrics(gamma_hat, p_hat)

    if verbose:
        print(f"\n  Results:")
        print(f"    avg D_KL(joint || PM)  = {metrics.kl_avg:.4e}  (target ~2.474e-3)")
        print(f"    MAE_PD^{{PM}}            = {metrics.mae_pd_pm:.4e}  (target ~2.193e-4)")
        print(f"\n  Borrower-level KL distribution:")
        print(f"    min  = {kl_values.min():.4e}")
        print(f"    p25  = {np.percentile(kl_values, 25):.4e}")
        print(f"    med  = {np.median(kl_values):.4e}")
        print(f"    p75  = {np.percentile(kl_values, 75):.4e}")
        print(f"    max  = {kl_values.max():.4e}")
        print(f"\n  PD distortion |hat_PD^PM - hat_PD| distribution:")
        pd_diff = np.abs(pd_pm - pd_joint)
        print(f"    min  = {pd_diff.min():.4e}")
        print(f"    p25  = {np.percentile(pd_diff, 25):.4e}")
        print(f"    med  = {np.median(pd_diff):.4e}")
        print(f"    p75  = {np.percentile(pd_diff, 75):.4e}")
        print(f"    max  = {pd_diff.max():.4e}")

    # ------------------------------------------------------------------
    # Step 2: Save metrics table
    # ------------------------------------------------------------------
    print("\n[2/3] Saving results...")
    _save_pm_metrics_table(metrics)

    # ------------------------------------------------------------------
    # Step 3: Save plots
    # ------------------------------------------------------------------
    _save_pm_plots(gamma_hat, q_pm, kl_values, pd_joint, pd_pm, metrics)

    results = {
        "metrics": metrics,
        "q_pm": q_pm,
        "kl_values": kl_values,
        "pd_joint": pd_joint,
        "pd_pm": pd_pm,
    }
    print("\nExperiment 2 complete.")
    return results


# ---------------------------------------------------------------------------
# Helper: save metrics table
# ---------------------------------------------------------------------------

def _save_pm_metrics_table(metrics) -> None:
    """Save product-of-marginals headline metrics to CSV."""
    rows = [
        {
            "Metric": "avg_KL_joint_vs_PM",
            "Description": "Average D_KL(joint || product-marginals)",
            "Achieved": f"{metrics.kl_avg:.4e}",
            "Target (paper)": "2.474e-3",
        },
        {
            "Metric": "MAE_PD_PM",
            "Description": "Average absolute PD distortion from factorization",
            "Achieved": f"{metrics.mae_pd_pm:.4e}",
            "Target (paper)": "2.193e-4",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp2_pm_metrics.csv", index=False)
    print(f"    Saved: {RESULTS_DIR / 'exp2_pm_metrics.csv'}")


# ---------------------------------------------------------------------------
# Helper: save plots
# ---------------------------------------------------------------------------

def _save_pm_plots(
    gamma: np.ndarray,
    q_pm: np.ndarray,
    kl_values: np.ndarray,
    pd_joint: np.ndarray,
    pd_pm: np.ndarray,
    metrics,
) -> None:
    """Save diagnostic plots for Experiment 2."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Histogram of borrower-level KL(joint || PM)
    ax = axes[0]
    ax.hist(kl_values, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(metrics.kl_avg, color="red", linestyle="--", lw=2,
               label=f"Mean = {metrics.kl_avg:.4e}")
    ax.set_xlabel("D_KL(joint || product-marginals)")
    ax.set_ylabel("Borrower Count")
    ax.set_title("Distribution of Posterior Factorization KL Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Joint PD vs PM PD
    ax = axes[1]
    ax.scatter(pd_joint, pd_pm, alpha=0.2, s=5, c="darkorange")
    lims = [min(pd_joint.min(), pd_pm.min()), max(pd_joint.max(), pd_pm.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="No distortion")
    ax.set_xlabel("PD from joint posterior (hat_PD_i)")
    ax.set_ylabel("PD from product-marginals (hat_PD_i^PM)")
    ax.set_title(f"PD Distortion from Factorization\nMAE={metrics.mae_pd_pm:.4e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-profile comparison of joint vs factorized posterior means
    ax = axes[2]
    profile_labels = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    joint_means = gamma.mean(axis=0)
    pm_means = q_pm.mean(axis=0)
    x = np.arange(4)
    width = 0.35
    bars1 = ax.bar(x - width/2, joint_means, width, label="Joint posterior mean",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, pm_means, width, label="Product-marginals mean",
                   color="darkorange", alpha=0.8)
    ax.set_xlabel("Latent Profile (Z_R, Z_F)")
    ax.set_ylabel("Mean Posterior Probability")
    ax.set_title("Mean Joint vs Product-Marginals Posterior by Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(profile_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Experiment 2: Product-of-Marginals Approximation Analysis\n"
        f"avg KL = {metrics.kl_avg:.4e} | avg |PD distortion| = {metrics.mae_pd_pm:.4e}",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_product_marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {RESULTS_DIR / 'exp2_product_marginals.png'}")


if __name__ == "__main__":
    # When run standalone, first execute Experiment 1 to get inputs
    from exp.exp1_recovery import run_exp1
    results_1 = run_exp1(verbose=False)
    run_exp2(
        gamma_hat=results_1["gamma_hat"],
        p_hat=results_1["p_hat"],
        verbose=True,
    )
