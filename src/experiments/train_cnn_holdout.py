"""
train_cnn_holdout.py — train CNN-LSTM while excluding the CPSC holdout.

Modes:
  cpsc       : controlled model trained only on CPSC complement.
  combined   : deployment model trained on CPSC complement + PhysioNet 2017.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import wfdb
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.experiments.provenance import read_json, write_json, write_run_metadata
from src.models.cnn_lstm import build_model


P17_CANDIDATES = [
    "data/raw/challenge_2017/training2017",
    "data/raw/challenge_2017",
    "data/raw/training2017",
]


def preprocess_signal(sig: np.ndarray, target_len: int = 5000) -> np.ndarray:
    sig = np.nan_to_num(sig.astype(np.float32))
    sig = np.clip(sig, -2.0, 2.0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    sig = np.clip(sig, -5.0, 5.0)
    if len(sig) >= target_len:
        return sig[:target_len]
    return np.pad(sig, (0, target_len - len(sig)))


class ArrayECGDataset(Dataset):
    def __init__(self, records: list[np.ndarray], labels: list[int], paths: list[str]):
        self.records = records
        self.labels = labels
        self.paths = paths

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.records[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_cpsc_records(manifest: dict[str, Any], split: str = "train") -> tuple[list[np.ndarray], list[int], list[str]]:
    rows, labels, paths = [], [], []
    for r in manifest["records"]:
        if r["split"] != split:
            continue
        try:
            record = wfdb.rdrecord(r["path"])
            leads = [n.strip().upper() for n in record.sig_name]
            if "I" not in leads:
                continue
            sig = record.p_signal[:, leads.index("I")]
            rows.append(preprocess_signal(sig))
            labels.append(int(r["label"]))
            paths.append(r["path"])
        except Exception:
            continue
    return rows, labels, paths


def add_physionet_2017(records: list[np.ndarray], labels: list[int], paths: list[str]) -> None:
    from src.preprocessing.ecg_dataset_combined import CombinedECGDataset

    p17_dir = next((p for p in P17_CANDIDATES if os.path.exists(p)), None)
    if not p17_dir:
        print("PhysioNet 2017 not found; combined mode will use CPSC complement only")
        return
    dummy_cpsc = "__no_cpsc_for_p17_only__"
    ds = CombinedECGDataset(dummy_cpsc, p17_dir)
    for rec, label, source in zip(ds.records, ds.labels, ds.sources):
        if source == "physionet2017":
            records.append(preprocess_signal(rec))
            labels.append(int(label))
            paths.append(f"physionet2017:{len(paths)}")


def evaluate(model, loader, device, threshold: float) -> dict[str, Any]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = torch.sigmoid(model(X).squeeze()).detach().cpu().numpy()
            probs.extend(np.atleast_1d(p).tolist())
            labels.extend(y.cpu().int().numpy().tolist())
    probs_arr = np.array(probs)
    labels_arr = np.array(labels, dtype=int)
    preds = (probs_arr >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels_arr, probs_arr)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds, zero_division=0)),
        "probs": probs_arr.tolist(),
        "labels": labels_arr.tolist(),
    }


def threshold_for_specificity(y_true: np.ndarray, probs: np.ndarray, specificity: float = 0.90) -> float:
    negatives = probs[y_true == 0]
    if len(negatives) == 0:
        return 0.4
    return float(np.quantile(negatives, specificity))


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = read_json(args.manifest)
    records, labels, paths = load_cpsc_records(manifest, split="train")
    if args.mode == "combined":
        add_physionet_2017(records, labels, paths)
    if len(np.unique(labels)) != 2:
        raise ValueError("CNN training set must contain both classes")

    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.validation_fraction, stratify=np.array(labels), random_state=args.seed
    )
    train_ds = ArrayECGDataset([records[i] for i in train_idx], [labels[i] for i in train_idx], [paths[i] for i in train_idx])
    val_ds = ArrayECGDataset([records[i] for i in val_idx], [labels[i] for i in val_idx], [paths[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_model(input_length=5000).to(device)
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = -1.0
    best_metrics: dict[str, Any] = {}
    no_improve = 0
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "cnn_lstm_cpsc_complement.pt" if args.mode == "cpsc" else "cnn_lstm_combined_deploy.pt"
    ckpt_path = processed_dir / ckpt_name
    legacy_ckpt_path = (
        processed_dir / "cnn_lstm_best.pt"
        if args.mode == "cpsc"
        else processed_dir / "cnn_lstm_combined_best.pt"
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = X + torch.randn_like(X) * args.noise_std
            loss = criterion(model(X).squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device, threshold=args.threshold)
        print(
            f"epoch={epoch+1} loss={np.mean(losses):.4f} "
            f"auc={metrics['auc']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = metrics
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    val_probs = np.array(best_metrics["probs"])
    val_labels = np.array(best_metrics["labels"], dtype=int)
    matched_threshold = threshold_for_specificity(val_labels, val_probs, args.matched_specificity)

    out_dir = write_run_metadata(extra={"stage": f"train_cnn_{args.mode}"})
    summary = {
        "model": f"CNN-LSTM {args.mode}",
        "manifest": str(args.manifest),
        "manifest_holdout_sha256": manifest["holdout_record_sha256"],
        "n_records_total_training_pool": len(labels),
        "n_train_records": len(train_ds),
        "n_validation_records": len(val_ds),
        "fixed_threshold": args.threshold,
        "matched_specificity": args.matched_specificity,
        "matched_specificity_threshold": matched_threshold,
        "best_validation_metrics": {k: v for k, v in best_metrics.items() if k not in ("probs", "labels")},
        "checkpoint_path": str(ckpt_path),
        "legacy_checkpoint_path": str(legacy_ckpt_path),
        "mode": args.mode,
    }
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, legacy_ckpt_path)
    write_json(out_dir / f"cnn_{args.mode}_training.json", summary)
    joblib.dump(summary, processed_dir / f"cnn_{args.mode}_thresholds.pkl")
    print(f"Saved CNN checkpoint -> {ckpt_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM excluding CPSC holdout")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["cpsc", "combined"], default="cpsc")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--matched-specificity", type=float, default=0.90)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
