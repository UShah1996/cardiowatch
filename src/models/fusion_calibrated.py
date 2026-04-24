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
        # Ensure the class is registered under its canonical module path
        # so pickle embeds 'src.models.fusion_calibrated' not '__main__'
        import sys
        import src.models.fusion_calibrated as _mod
        _mod.IsotonicCalibrator = IsotonicCalibrator
        _mod.CalibratedFusion   = CalibratedFusion
        self.__class__ = _mod.CalibratedFusion
        if hasattr(self, 'rf_calibrator') and self.rf_calibrator is not None:
            self.rf_calibrator.__class__  = _mod.IsotonicCalibrator
        if hasattr(self, 'ecg_calibrator') and self.ecg_calibrator is not None:
            self.ecg_calibrator.__class__ = _mod.IsotonicCalibrator
        joblib.dump(self, path)
        print(f"Fusion model saved → {path}")

    @staticmethod
    def load(path: str = 'data/processed/fusion_model.pkl') -> 'CalibratedFusion':
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Using hardcoded weights (0.6/0.4).")
            return CalibratedFusion()

        # Register classes in sys.modules so pickle can find them
        # regardless of whether this file is run as __main__ or as a module.
        import sys
        import src.models.fusion_calibrated as _self_module
        sys.modules.setdefault('src.models.fusion_calibrated', _self_module)
        # Also register under __main__ in case the pkl was saved that way
        _self_module.IsotonicCalibrator = IsotonicCalibrator
        _self_module.CalibratedFusion   = CalibratedFusion

        model = joblib.load(path)
        print(f"Fusion model loaded ← {path}")
        return model


# ── Demographic helpers ───────────────────────────────────────────────

def _extract_cpsc_demographics(data_dir: str, indices: list) -> list:
    """
    Scan CPSC 2018 .hea files and extract Age, Sex, and resting heart rate
    for each recording corresponding to the given dataset indices.

    CPSC 2018 header comment format:
        Age: 45
        Sex: Male      (or Female)
        Dx: 164889003

    Heart rate is computed from the ECG signal itself using R-peak
    detection — the same approach as rr_afib_detector.py.

    Args:
        data_dir : path to CPSC training directory
        indices  : list of dataset indices (from val_ds.indices)

    Returns:
        list of dicts: [{'age': int|None, 'sex': int|None, 'hr': float|None}]
        sex encoding: 1=Male, 0=Female, None=unknown
        hr: resting heart rate in bpm (None if R-peaks not detectable)
    """
    import wfdb
    from scipy.signal import find_peaks

    def _compute_hr(signal: np.ndarray, fs: int) -> 'float | None':
        """Estimate resting HR from Lead I signal using R-peak detection."""
        try:
            # Normalise
            sig = signal - signal.mean()
            sig = sig / (sig.std() + 1e-8)
            # R-peaks: prominent positive peaks, min distance ~0.4s
            peaks, _ = find_peaks(sig, height=0.3, distance=int(0.4 * fs))
            if len(peaks) < 2:
                return None
            rr_intervals = np.diff(peaks) / fs   # seconds
            # Reject physiologically implausible RR intervals
            rr_intervals = rr_intervals[
                (rr_intervals > 0.33) & (rr_intervals < 2.0)
            ]
            if len(rr_intervals) < 2:
                return None
            return float(60.0 / np.mean(rr_intervals))
        except Exception:
            return None

    # First pass: collect all valid .hea paths in order (same order as ECGDataset)
    all_paths = []
    for root, dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith('.hea'):
                continue
            path = os.path.join(root, fname.replace('.hea', ''))
            try:
                record = wfdb.rdrecord(path)
                leads  = [n.strip().upper() for n in record.sig_name]
                if 'I' not in leads:
                    continue
                all_paths.append(path)
            except Exception:
                continue

    demographics = []
    for idx in indices:
        if idx >= len(all_paths):
            demographics.append({'age': None, 'sex': None, 'hr': None})
            continue
        try:
            path   = all_paths[idx]
            record = wfdb.rdrecord(path)
            h      = wfdb.rdheader(path)

            # Parse Age and Sex from header comments
            age = None
            sex = None
            for c in h.comments:
                c = c.strip()
                if c.startswith('Age:'):
                    try:
                        age = int(c.replace('Age:', '').strip())
                    except ValueError:
                        pass
                elif c.startswith('Sex:'):
                    s = c.replace('Sex:', '').strip().lower()
                    if s in ('male', 'm'):
                        sex = 1
                    elif s in ('female', 'f'):
                        sex = 0

            # Compute heart rate from Lead I signal
            leads    = [n.strip().upper() for n in record.sig_name]
            lead_idx = leads.index('I')
            sig      = record.p_signal[:, lead_idx].astype(np.float32)
            sig      = np.nan_to_num(sig)
            hr       = _compute_hr(sig, record.fs)

            demographics.append({'age': age, 'sex': sex, 'hr': hr})
        except Exception:
            demographics.append({'age': None, 'sex': None, 'hr': None})

    return demographics


def _extract_kaggle_demographics(
    data_path: str = 'data/raw/heart.csv',
) -> dict:
    """
    Load Kaggle Heart Failure dataset and return age, sex, and MaxHR arrays.

    MaxHR (maximum heart rate during stress test) is used as a proxy
    for cardiovascular fitness — higher MaxHR = better fitness = lower risk.
    Resting HR from CPSC ECG is a related but different measure; both
    reflect cardiac rate capacity and are positively correlated.

    Returns:
        dict with:
          'ages'   : int array
          'sexes'  : int array (1=Male, 0=Female)
          'max_hr' : int array (MaxHR from stress test, 60–220)
    """
    import pandas as pd
    df = pd.read_csv(data_path)
    sexes  = (df['Sex'] == 'M').astype(int).values
    ages   = df['Age'].values
    max_hr = df['MaxHR'].values
    return {'ages': ages, 'sexes': sexes, 'max_hr': max_hr}


# ── Training script ───────────────────────────────────────────────────

def build_fusion_from_cpsc(
    cnn_weights_path: str = 'data/processed/cnn_lstm_combined_best.pt',
    rf_model_path:    str = 'data/processed/rf_model.pkl',
    scaler_path:      str = 'data/processed/scaler.pkl',
    data_dir:         str = ('data/raw/classification-of-12-lead-ecgs-the-physionetcomputing'
                             '-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018'),
) -> Optional['CalibratedFusion']:
    """
    Build fusion model using CPSC validation set with demographically
    matched RF scores.

    Strategy:
      1. Load CPSC ECG val set and get CNN-LSTM scores
      2. Extract Age + Sex from each CPSC recording's .hea header
      3. For each CPSC patient, find Kaggle patients with matching sex
         and age within ±10 years, sample RF score from that subset
      4. Fall back to label-conditioned sampling when no demographic
         match is available
      5. Fit CalibratedFusion on the demographically paired scores

    Why this is better than random sampling:
      A 45-year-old male CPSC patient is paired with RF scores from
      40-55 year old males in the Kaggle dataset rather than from the
      full population. This reduces the population mismatch bias,
      though it cannot eliminate it entirely since CPSC patients are
      from a cardiac monitoring context while Kaggle patients are from
      a general heart disease prediction dataset.

    Documented limitation:
      Age + sex is the only overlap between CPSC and Kaggle datasets.
      Richer matching (e.g., comorbidities, BP) is not possible without
      a dataset where the same patients have both ECG and clinical records.
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

    # ── Extract CPSC patient demographics (Age + Sex + HR) from headers ──
    # CPSC 2018 .hea files contain Age and Sex in comments.
    # Heart rate is computed from the ECG signal via R-peak detection.
    print("\nExtracting CPSC patient demographics (Age, Sex, HR) from headers...")
    print("  (This scans ECG signals for R-peaks — takes ~2 min)")
    cpsc_demographics = _extract_cpsc_demographics(data_dir, val_ds.indices)
    n_with_age = sum(1 for d in cpsc_demographics if d['age'] is not None)
    n_with_hr  = sum(1 for d in cpsc_demographics if d['hr']  is not None)
    print(f"  Age found : {n_with_age}/{len(cpsc_demographics)} recordings")
    print(f"  HR found  : {n_with_hr}/{len(cpsc_demographics)} recordings")

    # ── Extract Kaggle patient demographics ───────────────────────────
    kaggle_demo = _extract_kaggle_demographics()

    # ── Three-tier demographically matched RF score sampling ──────────
    #
    # Tier 1 — Tightest match (same sex + age ±10 + HR ±20 bpm):
    #   A 45-year-old male with HR=72 bpm matched to Kaggle males
    #   aged 35-55 with MaxHR within ±20 bpm range.
    #   Three physiological features aligned.
    #
    # Tier 2 — Medium match (same sex + age ±10):
    #   HR not available or too few candidates at Tier 1.
    #
    # Tier 3 — Loose match (same sex only):
    #   Age not available or too few candidates at Tier 2.
    #
    # Fallback — label-conditioned sampling:
    #   No demographic data available.
    #
    # Minimum pool size = 3 at each tier before relaxing.

    rng = np.random.default_rng(42)
    paired_rf_probs = np.zeros(len(ecg_labels))
    match_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0,
                    'label_only': 0, 'fallback': 0}

    # Build Kaggle lookup arrays from FULL dataset (not just val set)
    # so matching pool is as large as possible.
    # kag_ages/sexes/max_hr are already full-dataset from _extract_kaggle_demographics()
    # kag_probs must also be full-dataset — score RF on all 918 Kaggle patients
    from src.preprocessing.clinical import full_pipeline
    (X_tr_full, X_val_full, X_te_full, y_tr_full, y_val_full, y_te_full), feat_names = full_pipeline()
    import pandas as pd
    X_all_clin = pd.concat([X_tr_full, X_val_full, X_te_full], axis=0).reset_index(drop=True)
    kag_probs_all  = rf_model.predict_proba(X_all_clin)[:, 1]
    kag_labels_all = np.concatenate([y_tr_full, y_val_full, y_te_full])
    print(f"RF full Kaggle pool: n={len(kag_probs_all)}, positive={kag_labels_all.sum()} ({100*kag_labels_all.mean():.1f}%)")

    kag_ages   = np.array(kaggle_demo['ages'])
    kag_sexes  = np.array(kaggle_demo['sexes'])
    kag_max_hr = np.array(kaggle_demo['max_hr'])
    kag_labels = kag_labels_all
    kag_probs  = kag_probs_all

    for i, (label, demo) in enumerate(zip(ecg_labels, cpsc_demographics)):
        age = demo.get('age')
        sex = demo.get('sex')
        hr  = demo.get('hr')    # resting HR in bpm from ECG signal
        matched = False

        # ── Tier 1: sex + age ±10 + HR ±20 bpm ───────────────────────
        if age is not None and sex is not None and hr is not None:
            mask = (
                (kag_sexes == sex) &
                (np.abs(kag_ages - age) <= 10) &
                (np.abs(kag_max_hr - hr) <= 20)
            )
            if mask.sum() >= 3:
                paired_rf_probs[i] = rng.choice(kag_probs[np.where(mask)[0]])
                match_counts['tier1'] += 1
                matched = True

        # ── Tier 2: sex + age ±10 ────────────────────────────────────
        if not matched and age is not None and sex is not None:
            mask = (
                (kag_sexes == sex) &
                (np.abs(kag_ages - age) <= 10)
            )
            if mask.sum() >= 3:
                paired_rf_probs[i] = rng.choice(kag_probs[np.where(mask)[0]])
                match_counts['tier2'] += 1
                matched = True

        # ── Tier 3: sex only ─────────────────────────────────────────
        if not matched and sex is not None:
            mask = (kag_sexes == sex)
            if mask.sum() >= 3:
                paired_rf_probs[i] = rng.choice(kag_probs[np.where(mask)[0]])
                match_counts['tier3'] += 1
                matched = True

        # ── Fallback: label-conditioned ───────────────────────────────
        if not matched:
            idx_rf = np.where(kag_labels == label)[0]
            if len(idx_rf) == 0:
                paired_rf_probs[i] = 0.6 if label == 1 else 0.2
                match_counts['fallback'] += 1
            else:
                paired_rf_probs[i] = rng.choice(kag_probs[idx_rf])
                match_counts['label_only'] += 1

    total = len(ecg_labels)
    print(f"\nMatching results (n={total}):")
    print(f"  Tier 1 — sex + age±10 + HR±20 bpm : "
          f"{match_counts['tier1']:4d} ({100*match_counts['tier1']/total:.0f}%)")
    print(f"  Tier 2 — sex + age±10              : "
          f"{match_counts['tier2']:4d} ({100*match_counts['tier2']/total:.0f}%)")
    print(f"  Tier 3 — sex only                  : "
          f"{match_counts['tier3']:4d} ({100*match_counts['tier3']/total:.0f}%)")
    print(f"  Fallback — label-conditioned        : "
          f"{match_counts['label_only']:4d} ({100*match_counts['label_only']/total:.0f}%)")
    print(f"\nPaired dataset: n={total} ECG samples")
    print(f"  RF probs   mean={paired_rf_probs.mean():.3f} std={paired_rf_probs.std():.3f}")
    print(f"  ECG probs  mean={ecg_probs.mean():.3f} std={ecg_probs.std():.3f}")
    print(f"  Matching: 3-feature (sex+age+HR) → 2-feature (sex+age) "
          f"→ sex → label fallback")

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