"""
confidence_intervals.py — 95% Confidence Intervals for CardioWatch Metrics
============================================================================
Provides bootstrap and Wilson CI methods for all reported metrics.

Two modes:
  1. bootstrap_ci()    — for CNN-LSTM test-set metrics (single split, no CV)
                         Uses 2000 bootstrap resamples of the test set.
  2. cv_ci()           — for RF / XGBoost CV results
                         Uses the t-distribution across k fold scores.
  3. wilson_ci()       — for proportion metrics (accuracy, recall on small sets)
                         e.g. Apple Watch 34/36 or MIT-BIH window-level accuracy.

Why this matters:
  With only 92 clinical test patients, a ±1 recall difference could flip
  a result. Reporting mean ± CI forces the reader to see the uncertainty
  rather than treating a single number as ground truth.

Usage:
    python3 src/evaluation/confidence_intervals.py
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, precision_score
)
from typing import Tuple, Dict, Optional


# ── Bootstrap CI (for single-split models like CNN-LSTM) ──────────────

def bootstrap_ci(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    metric:     str   = 'auc',
    threshold:  float = 0.4,
    n_boot:     int   = 2000,
    ci:         float = 0.95,
    seed:       int   = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a single metric on a held-out set.

    Resamples (y_true, y_prob) with replacement n_boot times, computes
    the metric on each resample, and returns (point_estimate, lower, upper).

    Args:
        y_true    : ground-truth binary labels
        y_prob    : predicted probabilities (not thresholded)
        metric    : one of 'auc', 'recall', 'f1', 'precision'
        threshold : classification threshold (used for recall/f1/precision)
        n_boot    : number of bootstrap resamples (2000 is standard)
        ci        : confidence level (0.95 = 95% CI)
        seed      : random seed for reproducibility

    Returns:
        (point_estimate, lower_bound, upper_bound)

    Example:
        >>> auc, lo, hi = bootstrap_ci(y_true, y_prob, metric='auc')
        >>> print(f"AUC = {auc:.3f} (95% CI: {lo:.3f}–{hi:.3f})")
    """
    rng          = np.random.default_rng(seed)
    y_true       = np.array(y_true)
    y_prob       = np.array(y_prob)
    n            = len(y_true)
    boot_scores  = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        yp  = y_prob[idx]

        # Skip resamples where only one class is present (AUC undefined)
        if len(np.unique(yt)) < 2:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(yt, yp)
            elif metric == 'recall':
                score = recall_score(yt, (yp >= threshold).astype(int),
                                     zero_division=0)
            elif metric == 'f1':
                score = f1_score(yt, (yp >= threshold).astype(int),
                                 zero_division=0)
            elif metric == 'precision':
                score = precision_score(yt, (yp >= threshold).astype(int),
                                        zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            boot_scores.append(score)
        except Exception:
            continue

    boot_scores = np.array(boot_scores)
    alpha       = 1.0 - ci
    lower       = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper       = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    # Point estimate from full data
    y_pred = (y_prob >= threshold).astype(int)
    if metric == 'auc':
        point = float(roc_auc_score(y_true, y_prob))
    elif metric == 'recall':
        point = float(recall_score(y_true, y_pred, zero_division=0))
    elif metric == 'f1':
        point = float(f1_score(y_true, y_pred, zero_division=0))
    elif metric == 'precision':
        point = float(precision_score(y_true, y_pred, zero_division=0))

    return point, lower, upper


def bootstrap_all_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.4,
    n_boot:    int   = 2000,
    ci:        float = 0.95,
    seed:      int   = 42,
    label:     str   = 'Model',
) -> Dict[str, Tuple[float, float, float]]:
    """
    Run bootstrap CI for all four metrics at once.

    Returns dict: {'auc': (point, lo, hi), 'recall': ..., 'f1': ..., 'precision': ...}
    """
    metrics_out = {}
    print(f"\n{'='*60}")
    print(f"Bootstrap 95% CI ({n_boot} resamples) — {label}")
    print(f"  Test set size : {len(y_true)} samples")
    print(f"  Threshold     : {threshold}")
    print(f"  Positives     : {y_true.sum()} ({100*y_true.mean():.1f}%)")
    print(f"{'='*60}")

    for m in ['auc', 'recall', 'f1', 'precision']:
        pt, lo, hi = bootstrap_ci(
            y_true, y_prob,
            metric=m, threshold=threshold,
            n_boot=n_boot, ci=ci, seed=seed
        )
        metrics_out[m] = (pt, lo, hi)
        label_str      = m.upper().ljust(10)
        print(f"  {label_str}: {pt:.3f}  (95% CI: {lo:.3f} – {hi:.3f})")

    print(f"{'='*60}\n")
    return metrics_out


# ── CV CI (for RF / XGBoost with k-fold results) ─────────────────────

def cv_ci(
    fold_scores: np.ndarray,
    ci:          float = 0.95,
) -> Tuple[float, float, float]:
    """
    t-distribution confidence interval across k cross-validation fold scores.

    The standard ± std from cross_validate is a dispersion measure, not a
    proper CI. This uses the t-distribution with k-1 degrees of freedom,
    which is correct for small k (k=5 in our case).

    Args:
        fold_scores : array of per-fold metric values (length = n_folds)
        ci          : confidence level

    Returns:
        (mean, lower, upper)
    """
    n     = len(fold_scores)
    mean  = float(np.mean(fold_scores))
    se    = float(stats.sem(fold_scores))         # standard error of the mean
    t_crit = stats.t.ppf((1 + ci) / 2, df=n - 1) # two-tailed t critical value
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def cv_ci_report(
    results:    dict,
    model_name: str  = 'Model',
    ci:         float = 0.95,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Print full CI report from sklearn cross_validate results dict.

    Args:
        results    : output of sklearn.model_selection.cross_validate
        model_name : label for the report header
        ci         : confidence level

    Returns:
        dict: {'recall': (mean, lo, hi), 'f1': ..., 'auc': ...}
    """
    metrics_out = {}
    k           = len(results['test_recall'])

    print(f"\n{'='*60}")
    print(f"{k}-Fold CV — {ci*100:.0f}% CI (t-distribution, df={k-1}) — {model_name}")
    print(f"{'='*60}")

    mapping = {
        'recall'  : 'test_recall',
        'f1'      : 'test_f1',
        'auc'     : 'test_roc_auc',
    }

    for label, key in mapping.items():
        if key not in results:
            continue
        scores           = np.array(results[key])
        mean, lo, hi     = cv_ci(scores, ci=ci)
        metrics_out[label] = (mean, lo, hi)
        label_str          = label.upper().ljust(10)
        fold_str           = '  '.join(f'{s:.3f}' for s in scores)
        print(f"  {label_str}: {mean:.3f}  (95% CI: {lo:.3f} – {hi:.3f})")
        print(f"             Fold scores: [{fold_str}]")

    print(f"{'='*60}\n")
    return metrics_out


# ── Wilson CI (for proportion metrics on small counts) ────────────────

def wilson_ci(
    k:  int,
    n:  int,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Wilson score interval for a proportion k/n.

    Preferred over normal approximation for small n or extreme proportions
    (e.g., 34/36 = 0.944 — the normal approximation clips at 1.0).

    Args:
        k : number of successes (e.g., correct classifications)
        n : total trials
        ci: confidence level

    Returns:
        (proportion, lower, upper)

    Example:
        >>> p, lo, hi = wilson_ci(34, 36)
        >>> print(f"Apple Watch accuracy: {p:.1%} (95% CI: {lo:.1%}–{hi:.1%})")
        Apple Watch accuracy: 94.4% (95% CI: 81.3%–98.6%)
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    z   = stats.norm.ppf((1 + ci) / 2)
    p   = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return float(p), float(max(0, centre - half)), float(min(1, centre + half))


def wilson_report(counts: Dict[str, Tuple[int, int]], ci: float = 0.95) -> None:
    """
    Print Wilson CIs for multiple proportion metrics.

    Args:
        counts : dict of {label: (k_successes, n_total)}
        ci     : confidence level

    Example:
        wilson_report({
            'Apple Watch (Combined CNN-LSTM)': (34, 36),
            'Apple Watch (RR+RF Traditional)': (49, 54),
            'MIT-BIH (zero-shot)':             (25876, 28104),
        })
    """
    print(f"\n{'='*60}")
    print(f"Wilson Score {ci*100:.0f}% CI — Proportion Metrics")
    print(f"{'='*60}")
    for label, (k, n) in counts.items():
        p, lo, hi = wilson_ci(k, n, ci=ci)
        print(f"  {label}")
        print(f"    {k}/{n} = {p:.1%}  (95% CI: {lo:.1%} – {hi:.1%})")
    print(f"{'='*60}\n")


# ── Convenience: full project CI report ───────────────────────────────

def print_project_ci_summary(
    rf_cv_results:  Optional[dict] = None,
    xgb_cv_results: Optional[dict] = None,
) -> None:
    """
    Print the complete CI summary for all CardioWatch models.
    Pass in cross_validate result dicts from RF and XGBoost training.

    CNN-LSTM CIs are printed as placeholders — fill in after running
    train_cnn_lstm_cv.py which generates bootstrap CIs automatically.

    Apple Watch and MIT-BIH CIs use Wilson score (count-based).
    """
    print("\n" + "="*60)
    print("CARDIOWATCH — COMPLETE 95% CONFIDENCE INTERVAL REPORT")
    print("="*60)

    # ── Clinical models (CV-based) ────────────────────────────────────
    if rf_cv_results:
        cv_ci_report(rf_cv_results,  model_name='Random Forest (5-fold CV)')
    if xgb_cv_results:
        cv_ci_report(xgb_cv_results, model_name='XGBoost (5-fold CV)')

    # ── Real-world proportion metrics (Wilson score) ──────────────────
    wilson_report({
        'Apple Watch — Combined CNN-LSTM (34/36)' : (34, 36),
        'Apple Watch — RR+RF Traditional (49/54)' : (49, 54),
        'MIT-BIH AFib — CNN-LSTM zero-shot (AUC proxy: 25876/28104 windows)': (25876, 28104),
    })

    print("NOTE: CNN-LSTM test-set CIs are generated by train_cnn_lstm_cv.py")
    print("      Run that script first to populate bootstrap CI results.\n")


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running self-test with synthetic data...\n")

    rng    = np.random.default_rng(42)
    n      = 92        # matches your actual clinical test set size
    y_true = np.array([1]*23 + [0]*69)  # 25% positive rate
    # Simulate a good model
    y_prob = np.where(
        y_true,
        rng.beta(6, 2, n),
        rng.beta(2, 7, n),
    ).clip(0, 1)

    # ── Bootstrap CI (single split) ───────────────────────────────────
    bootstrap_all_metrics(
        y_true, y_prob, threshold=0.4, label='CNN-LSTM (synthetic test)'
    )

    # ── CV CI (simulated 5-fold) ──────────────────────────────────────
    synthetic_cv = {
        'test_recall' : np.array([0.91, 0.87, 0.93, 0.89, 0.90]),
        'test_f1'     : np.array([0.88, 0.85, 0.90, 0.87, 0.88]),
        'test_roc_auc': np.array([0.94, 0.93, 0.95, 0.94, 0.94]),
    }
    cv_ci_report(synthetic_cv, model_name='Random Forest (synthetic 5-fold)')

    # ── Wilson CI ─────────────────────────────────────────────────────
    wilson_report({
        'Apple Watch (34/36)' : (34, 36),
        'Apple Watch (49/54)' : (49, 54),
    })

    print("Self-test complete. Replace synthetic data with real results.")
    print("See random_forest.py and xgboost_model.py for integration.")
