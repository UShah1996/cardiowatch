"""
download_weights.py — Google Drive weight downloader for CardioWatch
Uses gdown which correctly handles Google Drive's virus-scan
confirmation page for all file types including .pt and .pkl.
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


def ensure_weights(status_callback=None) -> dict:
    """
    Download any missing model weights from Google Drive using gdown.
    Skips files that already exist and are non-empty (idempotent).
    """
    import gdown

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results = {}
    for name, fid in WEIGHTS.items():
        dest = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            results[name] = True
            continue

        if status_callback:
            status_callback(f"Downloading {name}...")

        try:
            url = f'https://drive.google.com/uc?id={fid}'
            gdown.download(url, dest, quiet=True, fuzzy=True)

            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                size_kb = os.path.getsize(dest) / 1024
                print(f"  ✓ {name} ({size_kb:.0f} KB)")
                results[name] = True
            else:
                print(f"  ✗ {name} — file too small, likely an error page")
                if os.path.exists(dest):
                    os.remove(dest)
                results[name] = False

        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            results[name] = False

    n_ok = sum(results.values())
    print(f"Weights ready: {n_ok}/{len(results)}")
    return results