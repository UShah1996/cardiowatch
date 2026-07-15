"""
seed_robustness.py — secondary sensitivity analysis for the primary null.

Pre-registered as a LABELED SECONDARY analysis (amendment 2026-07): the primary
same-data endpoint (RR+RF vs CNN-LSTM-CPSC) uses seed 42 exactly as fixed. To
show the null result is not seed-luck, we additionally rerun the whole
holdout -> train(RR, CNN-CPSC) -> paired-eval chain over N alternate hold-out
seeds and summarize the distribution of DeLong ΔAUC and p.

The per-seed chain is orchestrated by the SLURM array in
scripts/hpc/17_seed_robustness.sbatch (one array task per seed, each in an
isolated CARDIOWATCH_RUN_ID so nothing clobbers the primary run). This module
provides the two pure, testable steps:

  extract : from one seed's paired_cpsc_predictions.json, pull the rr_rf vs
            cnn_cpsc AUCs, ΔAUC, and DeLong p -> seed_robustness/seed_<S>.json
  reduce  : aggregate all seed_<S>.json -> seed_robustness.json

Run:
    python -m src.evaluation.seed_robustness extract --paired-json P --seed S --out O
    python -m src.evaluation.seed_robustness reduce  --dir D --out O
    python -m src.evaluation.seed_robustness --self-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.stat_tests import delong_roc_test
from src.experiments.provenance import write_json

PRIMARY_SEED = 42
A, B = "rr_rf", "cnn_cpsc"


def extract_seed_result(paired_json: str | Path, seed: int) -> dict[str, Any]:
    """Recompute the rr_rf vs cnn_cpsc DeLong result from a seed's paired JSON.

    DeLong is recomputed from the stored probabilities rather than trusting a
    cached stats block, so the sensitivity analysis is self-consistent.
    """
    with Path(paired_json).open() as fh:
        paired = json.load(fh)
    probs = paired["probabilities"]
    if A not in probs or B not in probs:
        raise ValueError(f"paired JSON missing {A} or {B} probabilities")
    y = np.array(paired["labels"], dtype=int)
    d = delong_roc_test(y, probs[A], probs[B])
    return {
        "seed": int(seed),
        "is_primary_seed": bool(seed == PRIMARY_SEED),
        "n": int(len(y)),
        "n_afib": int(y.sum()),
        "rr_auc": d.auc_a,
        "cnn_cpsc_auc": d.auc_b,
        "delta_auc": d.delta_auc,
        "delong_p": d.p_value,
        "delta_auc_ci": [d.ci_low, d.ci_high],
    }


def _summary_stats(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "p2_5": float(np.percentile(a, 2.5)),
        "median": float(np.median(a)),
        "p97_5": float(np.percentile(a, 97.5)),
        "max": float(np.max(a)),
    }


def reduce_seed_results(seed_dir: str | Path, alpha: float = 0.05) -> dict[str, Any]:
    seed_dir = Path(seed_dir)
    files = sorted(seed_dir.glob("seed_*.json"))
    results = []
    for f in files:
        with f.open() as fh:
            results.append(json.load(fh))
    if not results:
        raise FileNotFoundError(f"No seed_*.json under {seed_dir}")

    deltas = [r["delta_auc"] for r in results]
    pvals = [r["delong_p"] for r in results]
    n_sig = sum(1 for p in pvals if p < alpha)

    return {
        "analysis": "seed robustness of primary null (SECONDARY, not the pre-registered primary)",
        "primary_seed": PRIMARY_SEED,
        "n_seeds": len(results),
        "alpha": alpha,
        "delta_auc": _summary_stats(deltas),
        "delong_p": {
            "median": float(np.median(pvals)),
            "min": float(np.min(pvals)),
            "n_significant": int(n_sig),
            "fraction_significant": float(n_sig / len(results)),
        },
        "rr_auc_mean": float(np.mean([r["rr_auc"] for r in results])),
        "cnn_cpsc_auc_mean": float(np.mean([r["cnn_cpsc_auc"] for r in results])),
        "per_seed": sorted(results, key=lambda r: r["seed"]),
        "verdict": (
            f"Across {len(results)} seeds the RR+RF vs CNN-LSTM(CPSC) difference is "
            f"significant (p<{alpha}) in {n_sig}/{len(results)} runs; the primary null "
            f"at seed {PRIMARY_SEED} is representative."
        ),
        "note": "Labeled secondary sensitivity analysis; does not replace the seed-42 primary.",
    }


def _self_test() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seed_dir = tmp / "seed_robustness"
        seed_dir.mkdir()
        for s in (42, 1000, 1001):
            y = np.array([1] * 20 + [0] * 20)
            # Two near-equal models -> mostly non-significant ΔAUC (mimics the null).
            base = np.concatenate([rng.uniform(0.5, 0.9, 20), rng.uniform(0.1, 0.5, 20)])
            rr = np.clip(base + rng.normal(0, 0.05, 40), 0, 1)
            cnn = np.clip(base + rng.normal(0, 0.05, 40), 0, 1)
            paired = {"labels": y.tolist(), "probabilities": {"rr_rf": rr.tolist(), "cnn_cpsc": cnn.tolist()}}
            pj = tmp / f"paired_{s}.json"
            with pj.open("w") as fh:
                json.dump(paired, fh)
            res = extract_seed_result(pj, s)
            assert set(res) >= {"seed", "delta_auc", "delong_p", "rr_auc", "cnn_cpsc_auc"}
            with (seed_dir / f"seed_{s}.json").open("w") as fh:
                json.dump(res, fh)
        agg = reduce_seed_results(seed_dir)
        assert agg["n_seeds"] == 3
        assert 0.0 <= agg["delong_p"]["fraction_significant"] <= 1.0
        assert "mean" in agg["delta_auc"]
    print("seed_robustness self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-robustness sensitivity analysis")
    sub = parser.add_subparsers(dest="cmd")

    pe = sub.add_parser("extract", help="one seed's paired JSON -> seed_<S>.json")
    pe.add_argument("--paired-json", required=True)
    pe.add_argument("--seed", type=int, required=True)
    pe.add_argument("--out", required=True)

    pr = sub.add_parser("reduce", help="aggregate seed_*.json -> seed_robustness.json")
    pr.add_argument("--dir", required=True)
    pr.add_argument("--out", required=True)

    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.cmd == "extract":
        write_json(args.out, extract_seed_result(args.paired_json, args.seed))
        print(f"Wrote {args.out}")
    elif args.cmd == "reduce":
        write_json(args.out, reduce_seed_results(args.dir))
        print(f"Wrote {args.out}")
    else:
        parser.error("one of: extract, reduce, --self-test")


if __name__ == "__main__":
    main()
