"""
download_weights.py — Google Drive weight downloader for CardioWatch
=====================================================================
Downloads model weights from Google Drive at Streamlit Cloud startup.
Called once via @st.cache_resource — subsequent runs use cached files.

Usage (in app.py, before load_models()):
    from src.dashboard.download_weights import ensure_weights
    ensure_weights()
"""

import os
import requests

# ── File ID → local path mapping ─────────────────────────────────────
# Google Drive file IDs extracted from shareable links.
# All files are in My Drive > CardioWatch > weights (shared: anyone with link).

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
CHUNK_SIZE    = 32 * 1024  # 32 KB


def _gdrive_url(file_id: str) -> str:
    """Direct download URL for a publicly shared Google Drive file."""
    return f'https://drive.google.com/uc?export=download&id={file_id}'


def _download_file(filename: str, file_id: str) -> bool:
    """
    Download a single file from Google Drive to data/processed/.

    Handles Google's virus-scan confirmation page for files > ~100 KB
    by detecting the confirmation token and re-issuing the request.

    Returns True on success, False on failure.
    """
    dest = os.path.join(PROCESSED_DIR, filename)
    url  = _gdrive_url(file_id)

    try:
        session = requests.Session()
        response = session.get(url, stream=True, timeout=60)

        # Google redirects large files to a confirmation page
        # Detect by checking for the confirm token in response or cookies
        token = None
        for key, val in response.cookies.items():
            if key.startswith('download_warning'):
                token = val
                break

        if token:
            # Re-request with confirmation token
            response = session.get(
                url, params={'confirm': token},
                stream=True, timeout=120
            )

        # Also handle newer confirm mechanism via response URL
        if 'confirm' not in response.url and b'virus scan warning' in response.content[:500].lower():
            import re
            match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
            if match:
                response = session.get(
                    url, params={'confirm': match.group(1)},
                    stream=True, timeout=120
                )

        response.raise_for_status()

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        size_kb = os.path.getsize(dest) / 1024
        print(f"  ✓ {filename} ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"  ✗ {filename} failed: {e}")
        # Remove partial file if it exists
        if os.path.exists(dest):
            os.remove(dest)
        return False


def ensure_weights(status_callback=None) -> dict:
    """
    Download any missing model weights from Google Drive.

    Skips files that already exist locally (idempotent — safe to call
    on every startup; only downloads what's missing).

    Args:
        status_callback : optional callable(message) for Streamlit
                          st.status() or st.write() progress updates.

    Returns:
        dict mapping filename → True (present) / False (download failed)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results  = {}
    missing  = {
        name: fid for name, fid in WEIGHTS.items()
        if not os.path.exists(os.path.join(PROCESSED_DIR, name))
    }

    if not missing:
        if status_callback:
            status_callback("✅ All model weights already present.")
        return {name: True for name in WEIGHTS}

    if status_callback:
        status_callback(
            f"⬇️ Downloading {len(missing)} model weight file(s) "
            f"from Google Drive — first load takes ~30 seconds..."
        )

    for name, fid in WEIGHTS.items():
        dest = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(dest):
            results[name] = True
            continue
        if status_callback:
            status_callback(f"  Downloading {name}...")
        results[name] = _download_file(name, fid)

    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    if status_callback:
        msg = f"✅ Downloaded {n_ok}/{len(results)} weight files."
        if n_fail:
            msg += f" ⚠️ {n_fail} file(s) failed — app may fall back to demo mode."
        status_callback(msg)

    return results
