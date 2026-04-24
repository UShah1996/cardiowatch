# ── Weight download (Streamlit Cloud) ────────────────────────────────
# Place this BEFORE load_models() in app.py
# Replace the existing _download_weights() block entirely

import os

def _download_weights_on_startup():
    """
    Download model weights from Google Drive on startup.
    Runs synchronously — load_models() will not be called until this completes.
    Shows visible status in the Streamlit UI so you can see what's happening.
    """
    PROCESSED_DIR = 'data/processed'
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Check if weights already present — skip entirely if so
    if os.path.exists(f'{PROCESSED_DIR}/rf_model.pkl') and \
       os.path.getsize(f'{PROCESSED_DIR}/rf_model.pkl') > 10_000:
        return True  # already downloaded

    # Show download status in UI
    status_container = st.empty()
    status_container.info("⏳ Downloading model weights from Google Drive (first load only — ~2 min)...")

    try:
        import gdown
        from src.dashboard.download_weights import WEIGHTS

        progress = st.progress(0)
        results  = {}

        for i, (name, fid) in enumerate(WEIGHTS.items()):
            dest = os.path.join(PROCESSED_DIR, name)

            if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
                results[name] = True
                progress.progress((i + 1) / len(WEIGHTS))
                continue

            status_container.info(f"⏳ Downloading {name} ({i+1}/{len(WEIGHTS)})...")

            try:
                url = f'https://drive.google.com/uc?id={fid}'
                gdown.download(url, dest, quiet=False, fuzzy=True)

                size = os.path.getsize(dest) if os.path.exists(dest) else 0
                if size > 10_000:
                    results[name] = True
                else:
                    # File too small — likely an HTML error page from Drive
                    st.error(
                        f"❌ {name} downloaded but appears to be an error page "
                        f"({size} bytes). Check Google Drive sharing: "
                        f"the file must be shared as 'Anyone with the link'."
                    )
                    if os.path.exists(dest):
                        os.remove(dest)
                    results[name] = False

            except Exception as e:
                st.error(f"❌ Failed to download {name}: {e}")
                results[name] = False

            progress.progress((i + 1) / len(WEIGHTS))

        n_ok = sum(results.values())
        progress.empty()

        if n_ok == len(WEIGHTS):
            status_container.success(f"✅ All {n_ok} model weights downloaded successfully.")
            return True
        else:
            failed = [k for k, v in results.items() if not v]
            status_container.warning(
                f"⚠️ {n_ok}/{len(WEIGHTS)} weights downloaded. "
                f"Failed: {', '.join(failed)}. Running in partial/demo mode."
            )
            return n_ok > 0

    except ImportError:
        status_container.error(
            "❌ gdown not installed. Add 'gdown' to requirements.txt."
        )
        return False
    except Exception as e:
        status_container.error(f"❌ Weight download failed: {e}")
        return False


# Run before load_models() — blocks until complete
_weights_ready = _download_weights_on_startup()