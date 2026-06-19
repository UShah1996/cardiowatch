"""
stat_tests.py — paired statistical tests for CardioWatch evaluations.

Includes:
  - McNemar tests for paired fixed-threshold classifier correctness.
  - Fast DeLong paired ROC AUC test.
  - Holm correction for multiple pairwise tests.

The DeLong implementation is adapted from the widely used Yandex Data
School / Netflix VMAF implementation of Sun & Xu's fast DeLong method:

  Sun X, Xu W. Fast Implementation of DeLong's Algorithm for Comparing
  the Areas Under Correlated Receiver Operating Characteristic Curves.
  IEEE Signal Processing Letters. 2014;21(11):1389-1393.

Run:
    python3 src/evaluation/stat_tests.py --self-test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class McNemarResult:
    b: int
    c: int
    statistic: float | None
    p_value: float
    method: str
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float
    accuracy_diff: float
    n: int


@dataclass
class DeLongResult:
    auc_a: float
    auc_b: float
    delta_auc: float
    z: float
    p_value: float
    var_delta: float
    ci_low: float
    ci_high: float
    method: str = "paired DeLong"


def mcnemar_test(
    correct_a: Iterable[bool | int],
    correct_b: Iterable[bool | int],
    exact_threshold: int = 25,
) -> McNemarResult:
    """
    Paired McNemar test from two correctness vectors.

    b = A correct, B wrong.
    c = A wrong, B correct.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    bvec = np.asarray(list(correct_b), dtype=bool)
    if a.shape != bvec.shape:
        raise ValueError("correct_a and correct_b must have the same shape")

    b = int(np.sum(a & ~bvec))
    c = int(np.sum(~a & bvec))
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
        statistic = 0.0
        method = "exact binomial (no discordant pairs)"
    elif n_discordant <= exact_threshold:
        # Two-sided exact binomial under H0: P(b)=P(c)=0.5.
        p_value = float(2 * stats.binom.cdf(min(b, c), n_discordant, 0.5))
        p_value = min(p_value, 1.0)
        statistic = None
        method = "exact binomial"
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n_discordant)
        p_value = float(stats.chi2.sf(statistic, 1))
        method = "chi-square continuity-corrected"

    # Continuity-adjusted log OR CI is finite even when b or c is zero.
    b_adj = b + 0.5
    c_adj = c + 0.5
    odds_ratio = b_adj / c_adj
    se_log_or = np.sqrt(1.0 / b_adj + 1.0 / c_adj)
    ci_low = float(np.exp(np.log(odds_ratio) - 1.96 * se_log_or))
    ci_high = float(np.exp(np.log(odds_ratio) + 1.96 * se_log_or))
    accuracy_diff = float((np.sum(a) - np.sum(bvec)) / len(a))

    return McNemarResult(
        b=b,
        c=c,
        statistic=statistic,
        p_value=p_value,
        method=method,
        odds_ratio=float(odds_ratio),
        odds_ratio_ci_low=ci_low,
        odds_ratio_ci_high=ci_high,
        accuracy_diff=accuracy_diff,
        n=int(len(a)),
    )


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks for DeLong."""
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """
    Fast DeLong covariance for one or more classifiers.

    predictions_sorted_transposed shape: (n_classifiers, n_examples), with
    positives first and negatives second.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_roc_test(
    y_true: Iterable[int],
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    alpha: float = 0.05,
) -> DeLongResult:
    """Paired DeLong test for two correlated ROC AUCs."""
    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("y_true, scores_a, scores_b must have the same shape")
    if len(np.unique(y)) != 2:
        raise ValueError("DeLong requires both positive and negative labels")

    order = np.argsort(-y)
    label_1_count = int(np.sum(y == 1))
    preds = np.vstack([a, b])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    var_delta = float((contrast @ cov @ contrast.T).item())
    delta = float(aucs[0] - aucs[1])
    if var_delta <= 0:
        z = 0.0 if abs(delta) < 1e-12 else np.inf * np.sign(delta)
        p_value = 1.0 if z == 0.0 else 0.0
        ci_low = ci_high = delta
    else:
        z = float(delta / np.sqrt(var_delta))
        p_value = float(2 * stats.norm.sf(abs(z)))
        zcrit = stats.norm.ppf(1 - alpha / 2)
        half = float(zcrit * np.sqrt(var_delta))
        ci_low = delta - half
        ci_high = delta + half

    return DeLongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        delta_auc=delta,
        z=z,
        p_value=p_value,
        var_delta=var_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def holm_correction(p_values: Iterable[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in original order."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        adjusted_sorted[rank] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out.tolist()


def _self_test() -> None:
    # Perfectly identical predictors: p=1, delta=0.
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    same = delong_roc_test(y, s, s)
    assert abs(same.delta_auc) < 1e-12
    assert abs(same.p_value - 1.0) < 1e-12

    # AUC values should match sklearn exactly for the fixture.
    s2 = np.array([0.6, 0.55, 0.51, 0.49, 0.45, 0.4])
    res = delong_roc_test(y, s, s2)
    assert abs(res.auc_a - roc_auc_score(y, s)) < 1e-12
    assert abs(res.auc_b - roc_auc_score(y, s2)) < 1e-12

    mc = mcnemar_test([1, 1, 0, 0], [1, 0, 1, 0], exact_threshold=25)
    assert mc.b == 1 and mc.c == 1 and abs(mc.p_value - 1.0) < 1e-12

    holm = holm_correction([0.01, 0.04, 0.03])
    assert len(holm) == 3 and all(0 <= x <= 1 for x in holm)
    print("stat_tests self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioWatch statistical tests")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--paired-json", help="Paired prediction JSON from paired_cpsc_eval.py")
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.paired_json:
        parser.error("--paired-json or --self-test is required")

    with Path(args.paired_json).open() as f:
        paired = json.load(f)
    y = np.array(paired["labels"], dtype=int)
    names = paired["model_names"]
    outputs = {}
    p_values = []
    p_keys = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            key = f"{a_name}_vs_{b_name}"
            scores_a = paired["probabilities"][a_name]
            scores_b = paired["probabilities"][b_name]
            preds_a = paired["predictions_fixed"][a_name]
            preds_b = paired["predictions_fixed"][b_name]
            correct_a = np.array(preds_a, dtype=int) == y
            correct_b = np.array(preds_b, dtype=int) == y
            d = asdict(delong_roc_test(y, scores_a, scores_b))
            m = asdict(mcnemar_test(correct_a, correct_b))
            outputs[key] = {"delong": d, "mcnemar_fixed": m}
            p_keys.extend([(key, "delong"), (key, "mcnemar_fixed")])
            p_values.extend([d["p_value"], m["p_value"]])

    adjusted = holm_correction(p_values)
    for (key, test_name), adj in zip(p_keys, adjusted):
        outputs[key][test_name]["p_value_holm"] = adj

    out_path = Path(args.out) if args.out else Path(args.paired_json).with_name("stat_tests.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(outputs, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
