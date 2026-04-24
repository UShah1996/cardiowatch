"""
download_weights.py — Google Drive weight downloader for CardioWatch
Pure Python — no Streamlit dependency.
Called from app.py which handles all UI feedback.
"""

import os

WEIGHTS = {
    'cnn_lstm_combined_best.pt': '1iB6P4s6Gkgf3x2L1_9tcW6jExssLsNzA',
    'cnn_lstm_cv_best.pt':       '1boR7-dcItAgIRL2w8LgHfjnwrTBgSNj6',
    'fusion_model.pkl':          '1H060iL9aiH2e-7ocOo8xR1DeWIgXUbYx',
    'rf_model.pkl':              '1EYmVToWFHujQIfK34Bsr6DdCrsskTycL',
    'rr_rf_model.pkl':           '18Vci8UkVERR8yBvZpcwW0CGgfjYHDv1C',
    'scaler.pkl':                '1R2a79B2VEVAgvurDrWE4Xw1oXwfReEhn',
    'xgb_model.pkl':             '17WakvbrNXUR8bnhrheWV4XSoSh5mzdcS',
}

PROCESSED_DIR = 'data/processed'


def ensure_weights(log_fn=print) -> dict:
    """
    Download any missing model weights from Google Drive using gdown.
    Skips files that already exist and are non-empty (idempotent).

    Args:
        log_fn: callable for progress messages (default: print)
                pass a lambda or st.write for UI feedback

    Returns:
        dict {filename: True/False} — True if file is ready
    """
    try:
        import gdown
    except ImportError:
        log_fn("gdown not installed — add 'gdown' to requirements.txt")
        return {name: False for name in WEIGHTS}

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = {}

    for name, fid in WEIGHTS.items():
        dest = os.path.join(PROCESSED_DIR, name)

        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            log_fn(f"✓ {name} already present")
            results[name] = True
            continue

        log_fn(f"Downloading {name}...")
        try:
            url = f'https://drive.google.com/uc?id={fid}'
            gdown.download(url, dest, quiet=False)

            size = os.path.getsize(dest) if os.path.exists(dest) else 0
            if size > 10_000:
                log_fn(f"✓ {name} ({size // 1024} KB)")
                results[name] = True
            else:
                log_fn(
                    f"✗ {name} only {size} bytes — "
                    f"Google Drive permission error. "
                    f"Set sharing to 'Anyone with the link'."
                )
                if os.path.exists(dest):
                    os.remove(dest)
                results[name] = False

        except Exception as e:
            log_fn(f"✗ {name} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            results[name] = False

    n_ok = sum(results.values())
    log_fn(f"Weights ready: {n_ok}/{len(results)}")
    return results