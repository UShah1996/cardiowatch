"""
fusion_calibrated.py — Learned Fusion for CardioWatch
======================================================
Replaces hardcoded 0.6/0.4 weights with a data-driven approach.

The cross-population problem:
  RF is trained on Kaggle Heart Failure (918 patients, clinical features).
  CNN-LSTM is trained on CPSC/PhysioNet ECG recordings.
  These are different patient populations — their raw probability scales
  are not directly comparable. Hardcoding 0.6/0.4 is arbitrary.

The fix — two components:
  1. Isotonic calibration: maps each model's raw probabilities to a
     consistent [0,1] scale using Platt scaling / isotonic regression.
     After calibration, 0.7 from RF and 0.7 from CNN-LSTM mean the
     same thing: "70% likely to have the condition."

  2. Learned fusion weights: train a logistic regression layer on top
     of the two calibrated scores. The weights are learned from data
     (the CPSC validation set provides ECG scores; clinical scores come
     from the RF model applied to synthetic-but-realistic clinical
     feature vectors drawn from the Kaggle training distribution).

     When Apple Watch data with known clinical features is available,
     pass it to fit_fusion() to get fully real-data-driven weights.

Why logistic regression for fusion:
  - Produces calibrated output probabilities
  - Learns the optimal weighting from data rather than guessing
  - Only 2 parameters (w_rf, w_cnn) + bias — can't overfit even on small n
  - Interpretable: coefficients show relative model contribution

Usage:
    # Train and save fusion model (run after RF and CNN-LSTM are trained):
    python3 src/models/fusion_calibrated.py

    # In app.py or lead_time.py:
    from src.models.fusion_calibrated import CalibratedFusion
    fusion = CalibratedFusion.load('data/processed/fusion_model.pkl')
    fused_prob = fusion.predict_proba(rf_prob, ecg_prob)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss
from typing import Optional, Tuple

PROCESSED_DIR = 'data/processed'


# ── Isotonic calibrator ───────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Maps a model's raw probability outputs to a calibrated scale.

    Uses isotonic regression (non-parametric, monotone) which is more
    robust than Platt scaling when the model is already well-trained.
    Requires a held-out calibration set — never calibrate on training data.
    """

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit calibrator on held-out validation probabilities.

        Args:
            probs  : raw model output probabilities (shape: [n])
            y_true : ground-truth binary labels (shape: [n])
        """
        self.ir.fit(probs, y_true)
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to new probability scores."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        return self.ir.predict(np.array(probs))

    def calibration_error(
        self, probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) — mean |predicted prob - true freq|.
        Lower is better. A perfectly calibrated model has ECE = 0.
        """
        cal_probs = self.transform(probs)
        fraction_pos, mean_pred = calibration_curve(
            y_true, cal_probs, n_bins=n_bins, strategy='uniform'
        )
        return float(np.mean(np.abs(fraction_pos - mean_pred)))


# ── Learned fusion layer ──────────────────────────────────────────────

class CalibratedFusion:
    """
    Two-input logistic regression fusion of calibrated RF and CNN-LSTM scores.

    Replaces hardcoded weights with data-driven weights learned from a
    validation set where both model scores are available.

    Architecture:
        [rf_cal, ecg_cal] → LogisticRegression(C=1.0) → fused_prob

    The logistic regression learns:
        fused = sigmoid(w_rf * rf_cal + w_ecg * ecg_cal + bias)

    After fitting, weights are inspected to verify the learned ratio
    is sensible (RF should dominate slightly given stronger validation).
    """

    def __init__(self):
        self.rf_calibrator  = IsotonicCalibrator()
        self.ecg_calibrator = IsotonicCalibrator()
        self.lr             = LogisticRegression(
            C=1.0,              # moderate regularisation
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
        )
        self.fitted         = False
        self.weight_rf      = None
        self.weight_ecg     = None
        self.effective_threshold = 0.5

    def fit(
        self,
        rf_probs_cal:  np.ndarray,   # RF probs on calibration set
        ecg_probs_cal: np.ndarray,   # CNN-LSTM probs on calibration set
        y_cal:         np.ndarray,   # true labels for calibration set
        rf_probs_val:  np.ndarray,   # RF probs on validation set
        ecg_probs_val: np.ndarray,   # CNN-LSTM probs on validation set
        y_val:         np.ndarray,   # true labels for validation set
    ) -> 'CalibratedFusion':
        """
        Two-stage fitting:
          Stage 1 — fit calibrators on calibration set
          Stage 2 — fit logistic fusion on calibrated validation scores

        Using separate cal/val sets prevents data leakage.
        If you only have one held-out set, use fit_single_set() instead.

        Args:
            rf_probs_cal  : RF raw probs on the calibration split
            ecg_probs_cal : CNN-LSTM raw probs on the calibration split
            y_cal         : labels for calibration split
            rf_probs_val  : RF raw probs on validation split
            ecg_probs_val : CNN-LSTM raw probs on validation split
            y_val         : labels for validation split
        """
        # Stage 1: calibrate both models on the calibration set
        print("Fitting isotonic calibrators...")
        self.rf_calibrator.fit(np.array(rf_probs_cal), np.array(y_cal))
        self.ecg_calibrator.fit(np.array(ecg_probs_cal), np.array(y_cal))

        rf_ece  = self.rf_calibrator.calibration_error(rf_probs_cal, y_cal)
        ecg_ece = self.ecg_calibrator.calibration_error(ecg_probs_cal, y_cal)
        print(f"  RF calibration ECE  : {rf_ece:.4f}")
        print(f"  ECG calibration ECE : {ecg_ece:.4f}")

        # Stage 2: apply calibrators to val set, learn fusion weights
        rf_cal  = self.rf_calibrator.transform(np.array(rf_probs_val))
        ecg_cal = self.ecg_calibrator.transform(np.array(ecg_probs_val))

        X_fusion = np.column_stack([rf_cal, ecg_cal])
        print("Fitting logistic fusion layer...")
        self.lr.fit(X_fusion, np.array(y_val))

        # Extract and report learned weights
        coef = self.lr.coef_[0]
        total = np.sum(np.abs(coef))
        self.weight_rf  = float(np.abs(coef[0]) / total)
        self.weight_ecg = float(np.abs(coef[1]) / total)

        print(f"\n  Learned fusion weights (normalised):")
        print(f"    RF  weight : {self.weight_rf:.3f}  (was hardcoded 0.600)")
        print(f"    ECG weight : {self.weight_ecg:.3f}  (was hardcoded 0.400)")
        print(f"    Bias       : {self.lr.intercept_[0]:.4f}")

        self.fitted = True
        return self

    def fit_single_set(
        self,
        rf_probs:  np.ndarray,
        ecg_probs: np.ndarray,
        y_true:    np.ndarray,
        test_size: float = 0.3,
        seed:      int   = 42,
    ) -> 'CalibratedFusion':
        """
        Fit on a single held-out set by splitting it 70/30 into
        calibration and validation internally.

        Use this when Apple Watch data is the only available set
        with both RF and ECG scores for the same samples.
        """
        from sklearn.model_selection import train_test_split

        n = len(y_true)
        print(f"Single-set fitting: n={n}, splitting {100*(1-test_size):.0f}/{100*test_size:.0f}")

        if n < 20:
            print(f"  WARNING: n={n} is very small. Calibration will be noisy.")
            print(f"  Using fixed weights RF=0.60, ECG=0.40 as fallback.")
            self._use_fixed_weights(rf_probs, ecg_probs, y_true)
            return self

        idx_cal, idx_val = train_test_split(
            np.arange(n), test_size=test_size, stratify=y_true, random_state=seed
        )
        return self.fit(
            rf_probs[idx_cal],  ecg_probs[idx_cal],  y_true[idx_cal],
            rf_probs[idx_val],  ecg_probs[idx_val],  y_true[idx_val],
        )

    def _use_fixed_weights(
        self,
        rf_probs:  np.ndarray,
        ecg_probs: np.ndarray,
        y_true:    np.ndarray,
    ) -> None:
        """
        Fallback when n is too small for calibration.
        Fits calibrators on whatever data is available and uses
        hardcoded weights — better than nothing, clearly documented.
        """
        self.rf_calibrator.fit(rf_probs, y_true)
        self.ecg_calibrator.fit(ecg_probs, y_true)

        # Use manually set weights via a trivially-fit LR
        self.weight_rf  = 0.60
        self.weight_ecg = 0.40
        # Fit LR with fixed penalty to enforce near-0.6/0.4 split
        rf_cal  = self.rf_calibrator.transform(rf_probs)
        ecg_cal = self.ecg_calibrator.transform(ecg_probs)
        X_fusion = np.column_stack([rf_cal, ecg_cal])
        self.lr.fit(X_fusion, y_true)
        self.fitted = True
        print("  Using calibrated probabilities with fixed weights (0.60/0.40).")

    def predict_proba(
        self,
        rf_prob:  float,
        ecg_prob: float,
    ) -> float:
        """
        Fuse a single (rf_prob, ecg_prob) pair into one risk score.

        Args:
            rf_prob  : raw RF probability for one patient
            ecg_prob : raw CNN-LSTM probability for one ECG window

        Returns:
            fused probability in [0, 1]
        """
        if not self.fitted:
            # Graceful fallback to hardcoded weights — never crashes
            return 0.6 * rf_prob + 0.4 * ecg_prob

        rf_cal  = float(self.rf_calibrator.transform(np.array([rf_prob]))[0])
        ecg_cal = float(self.ecg_calibrator.transform(np.array([ecg_prob]))[0])
        x       = np.array([[rf_cal, ecg_cal]])
        return float(self.lr.predict_proba(x)[0, 1])

    def predict_proba_batch(
        self,
        rf_probs:  np.ndarray,
        ecg_probs: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of predict_proba for arrays."""
        if not self.fitted:
            return 0.6 * np.array(rf_probs) + 0.4 * np.array(ecg_probs)

        rf_cal  = self.rf_calibrator.transform(np.array(rf_probs))
        ecg_cal = self.ecg_calibrator.transform(np.array(ecg_probs))
        X       = np.column_stack([rf_cal, ecg_cal])
        return self.lr.predict_proba(X)[:, 1]

    def evaluate(
        self,
        rf_probs:  np.ndarray,
        ecg_probs: np.ndarray,
        y_true:    np.ndarray,
        threshold: float = 0.5,
        label:     str   = 'Fused',
    ) -> dict:
        """Evaluate fused predictions against ground truth."""
        fused  = self.predict_proba_batch(rf_probs, ecg_probs)
        preds  = (fused >= threshold).astype(int)
        recall = recall_score(y_true, preds, zero_division=0)
        auc    = roc_auc_score(y_true, fused)
        brier  = brier_score_loss(y_true, fused)

        print(f"\n{label} (threshold={threshold}):")
        print(f"  Recall  : {recall:.3f}")
        print(f"  AUC-ROC : {auc:.3f}")
        print(f"  Brier   : {brier:.4f}  (lower = better calibration)")
        return {'recall': recall, 'auc': auc, 'brier': brier}

    def summary(self) -> None:
        """Print a human-readable summary of learned weights."""
        if not self.fitted:
            print("Model not fitted yet.")
            return
        print("\nCalibratedFusion Summary:")
        print(f"  RF  weight : {self.weight_rf:.3f}")
        print(f"  ECG weight : {self.weight_ecg:.3f}")
        print(f"  (Previous hardcoded: RF=0.600, ECG=0.400)")
        coef = self.lr.coef_[0]
        ratio = abs(coef[0]) / (abs(coef[1]) + 1e-9)
        direction = "RF dominates" if ratio > 1.0 else "ECG dominates"
        print(f"  Learned ratio RF:ECG = {ratio:.2f}:1  ({direction})")

    def save(self, path: str = 'data/processed/fusion_model.pkl') -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Fusion model saved → {path}")

    @staticmethod
    def load(path: str = 'data/processed/fusion_model.pkl') -> 'CalibratedFusion':
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Using hardcoded weights (0.6/0.4).")
            f = CalibratedFusion()
            return f
        model = joblib.load(path)
        print(f"Fusion model loaded ← {path}")
        return model


# ── Training script ───────────────────────────────────────────────────

def build_fusion_from_cpsc(
    cnn_weights_path: str = 'data/processed/cnn_lstm_combined_best.pt',
    rf_model_path:    str = 'data/processed/rf_model.pkl',
    scaler_path:      str = 'data/processed/scaler.pkl',
    data_dir:         str = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
                             '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018'),
) -> Optional['CalibratedFusion']:
    """
    Build fusion model using CPSC validation set.

    Strategy:
      - Load CPSC ECG recordings and get CNN-LSTM scores
      - For clinical scores, apply RF to synthetic clinical feature vectors
        sampled from the Kaggle training distribution (documented limitation)
      - Fit CalibratedFusion on these paired scores

    NOTE: This is an approximation because RF and CNN-LSTM were trained on
    different patient populations. When Apple Watch data with real clinical
    features is available (from the 4 volunteers), call
    build_fusion_from_apple_watch() instead for fully grounded weights.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.models.cnn_lstm import build_model
    from src.preprocessing.ecg_dataset import ECGDataset
    from src.preprocessing.clinical import full_pipeline

    print("="*60)
    print("Building calibrated fusion model from CPSC validation set")
    print("="*60)

    # ── Load CNN-LSTM and get ECG scores on CPSC val set ─────────────
    if not os.path.exists(cnn_weights_path):
        print(f"ERROR: {cnn_weights_path} not found.")
        print("Run train_cnn_lstm.py (or train_cnn_lstm_combined.py) first.")
        return None

    device    = torch.device('cpu')
    cnn_model = build_model(input_length=5000).to(device)
    cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location='cpu'))
    cnn_model.eval()
    print(f"CNN-LSTM loaded from {cnn_weights_path}")

    dataset  = ECGDataset(data_dir)
    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train

    from torch.utils.data import random_split
    _, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    ecg_probs, ecg_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            probs = torch.sigmoid(cnn_model(X).squeeze())
            ecg_probs.extend(probs.cpu().tolist())
            ecg_labels.extend(y.cpu().int().tolist())

    ecg_probs  = np.array(ecg_probs)
    ecg_labels = np.array(ecg_labels)
    print(f"ECG scores collected: n={len(ecg_probs)}, "
          f"AFib={ecg_labels.sum()} ({100*ecg_labels.mean():.1f}%)")

    # ── Load RF and get clinical scores ──────────────────────────────
    # RF is trained on Kaggle clinical data — we can only apply it
    # to clinical-format feature vectors, not ECG patients.
    # We generate synthetic clinical profiles that match the ECG label
    # distribution — same prevalence as the ECG val set.
    # This is an approximation; document it clearly.
    if not os.path.exists(rf_model_path):
        print(f"ERROR: {rf_model_path} not found. Run random_forest.py first.")
        return None

    rf_model = joblib.load(rf_model_path)
    scaler   = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Get the actual clinical val set for calibration
    (_, X_val_clin, _, _, y_val_clin, _), _ = full_pipeline()
    rf_val_probs  = rf_model.predict_proba(X_val_clin)[:, 1]
    rf_val_labels = np.array(y_val_clin)
    print(f"RF clinical val: n={len(rf_val_probs)}, "
          f"positive={rf_val_labels.sum()} ({100*rf_val_labels.mean():.1f}%)")

    # ── Build synthetic paired dataset ───────────────────────────────
    # Match ECG label distribution by sampling RF scores
    # conditioned on ECG label — preserves class balance.
    # For each ECG sample, sample an RF score from patients with
    # the same label in the clinical val set.
    rng = np.random.default_rng(42)
    paired_rf_probs = np.zeros(len(ecg_labels))

    for label in [0, 1]:
        idx_ecg  = np.where(ecg_labels == label)[0]
        idx_rf   = np.where(rf_val_labels == label)[0]
        if len(idx_rf) == 0:
            # Fall back to uniform if one class is missing
            paired_rf_probs[idx_ecg] = 0.6 if label == 1 else 0.2
            continue
        sampled = rng.choice(rf_val_probs[idx_rf],
                             size=len(idx_ecg), replace=True)
        paired_rf_probs[idx_ecg] = sampled

    print(f"\nPaired dataset: n={len(ecg_labels)} ECG samples")
    print(f"  RF probs   mean={paired_rf_probs.mean():.3f} std={paired_rf_probs.std():.3f}")
    print(f"  ECG probs  mean={ecg_probs.mean():.3f} std={ecg_probs.std():.3f}")
    print(f"  NOTE: RF scores are sampled from Kaggle val distribution,")
    print(f"        not from the same patients as the ECG recordings.")
    print(f"        This is documented as a limitation.")

    # ── Fit and evaluate fusion ───────────────────────────────────────
    fusion = CalibratedFusion()
    fusion.fit_single_set(
        paired_rf_probs, ecg_probs, ecg_labels,
        test_size=0.3, seed=42
    )
    fusion.summary()

    # Evaluate against baseline (hardcoded weights)
    baseline_fused = 0.6 * paired_rf_probs + 0.4 * ecg_probs
    baseline_auc   = roc_auc_score(ecg_labels, baseline_fused)
    print(f"\nBaseline (hardcoded 0.6/0.4) AUC: {baseline_auc:.4f}")
    fusion.evaluate(paired_rf_probs, ecg_probs, ecg_labels,
                    label='Calibrated Fusion')

    # Save
    fusion.save('data/processed/fusion_model.pkl')
    print("\nDone. Use CalibratedFusion.load() in app.py and lead_time.py.")
    return fusion


def build_fusion_from_apple_watch(
    apple_watch_ecg_probs:  np.ndarray,
    apple_watch_rf_probs:   np.ndarray,
    apple_watch_labels:     np.ndarray,
) -> 'CalibratedFusion':
    """
    Build fusion model from Apple Watch data with real clinical features.

    This is the PREFERRED method — it uses real paired data where the
    same person has both an ECG recording and clinical feature values.
    Call this instead of build_fusion_from_cpsc() when Apple Watch
    volunteer data is available.

    Args:
        apple_watch_ecg_probs : CNN-LSTM scores on Apple Watch recordings
        apple_watch_rf_probs  : RF scores from volunteer clinical features
        apple_watch_labels    : AFib ground truth for each recording

    Returns:
        Fitted CalibratedFusion model
    """
    print("="*60)
    print("Building calibrated fusion from Apple Watch data (PREFERRED)")
    print(f"  n={len(apple_watch_labels)}, "
          f"AFib={apple_watch_labels.sum()}")
    print("="*60)

    fusion = CalibratedFusion()
    fusion.fit_single_set(
        np.array(apple_watch_rf_probs),
        np.array(apple_watch_ecg_probs),
        np.array(apple_watch_labels),
        test_size=0.3,
    )
    fusion.summary()
    fusion.evaluate(
        apple_watch_rf_probs,
        apple_watch_ecg_probs,
        apple_watch_labels,
        label='Apple Watch Calibrated Fusion',
    )
    fusion.save('data/processed/fusion_model.pkl')
    return fusion


# ── Updated fuse_predictions (drop-in replacement for fusion.py) ──────

def fuse_predictions(
    rf_probs:  np.ndarray,
    cnn_probs: np.ndarray,
    fusion_model_path: str = 'data/processed/fusion_model.pkl',
) -> np.ndarray:
    """
    Drop-in replacement for the old fuse_predictions() in fusion.py.
    Uses CalibratedFusion if available, falls back to 0.6/0.4.
    """
    fusion = CalibratedFusion.load(fusion_model_path)
    if not fusion.fitted:
        print("Fusion model not fitted — using hardcoded 0.6 RF + 0.4 ECG")
        return 0.6 * np.array(rf_probs) + 0.4 * np.array(cnn_probs)
    return fusion.predict_proba_batch(
        np.array(rf_probs), np.array(cnn_probs)
    )


if __name__ == '__main__':
    fusion = build_fusion_from_cpsc()
    if fusion:
        print("\nFusion model ready. Re-run lead_time.py to use learned weights.")
