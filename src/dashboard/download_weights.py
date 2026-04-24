"""
download_weights.py — Hugging Face Hub weight downloader for CardioWatch
No download limits, no virus-scan quotas, designed for ML model weights.

Repo: https://huggingface.co/UShah1996/cardiowatch-weights
"""

import os

REPO_ID       = 'UShah1996/cardiowatch-weights'
PROCESSED_DIR = 'data/processed'

WEIGHTS = [
    'cnn_lstm_combined_best.pt',
    'cnn_lstm_cv_best.pt',
    'fusion_model.pkl',
    'rf_model.pkl',
    'rr_rf_model.pkl',
    'scaler.pkl',
    'xgb_model.pkl',
]


def ensure_weights(log_fn=print) -> dict:
    """
    Download any missing model weights from Hugging Face Hub.
    Skips files that already exist and are non-empty (idempotent).

    Args:
        log_fn: callable for progress messages (default: print)
                pass a lambda or st.write for UI feedback

    Returns:
        dict {filename: True/False} — True if file is ready
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log_fn("huggingface_hub not installed — add 'huggingface_hub' to requirements.txt")
        return {name: False for name in WEIGHTS}

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = {}

    for name in WEIGHTS:
        dest = os.path.join(PROCESSED_DIR, name)

        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            log_fn(f'✓ {name} already present')
            results[name] = True
            continue

        log_fn(f'Downloading {name}...')
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                local_dir=PROCESSED_DIR,
                local_dir_use_symlinks=False,
            )
            size = os.path.getsize(dest) if os.path.exists(dest) else 0
            if size > 500:
                log_fn(f'✓ {name} ({size // 1024} KB)')
                results[name] = True
            else:
                log_fn(f'✗ {name} — downloaded but too small ({size} bytes)')
                if os.path.exists(dest):
                    os.remove(dest)
                results[name] = False

        except Exception as e:
            log_fn(f'✗ {name} failed: {e}')
            if os.path.exists(dest):
                os.remove(dest)
            results[name] = False

    n_ok = sum(results.values())
    log_fn(f'Weights ready: {n_ok}/{len(results)}')
    return results