"""
provenance.py — shared run/provenance helpers for paper experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RESULTS_ROOT = Path("docs/results")


def git_sha(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return "unknown"


def default_run_id() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{today}-{git_sha(short=True)}"


def get_run_id() -> str:
    return os.environ.get("CARDIOWATCH_RUN_ID", default_run_id())


def results_dir(run_id: str | None = None, create: bool = True) -> Path:
    path = RESULTS_ROOT / (run_id or get_run_id())
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_items(items: Iterable[Any]) -> str:
    payload = "\n".join(str(x) for x in items)
    return sha256_text(payload)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            ["python", "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def system_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(short=False),
        "git_short_sha": git_sha(short=True),
        "run_id": get_run_id(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "env": {
            "CARDIOWATCH_RUN_ID": os.environ.get("CARDIOWATCH_RUN_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
            "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def write_run_metadata(run_id: str | None = None, extra: dict[str, Any] | None = None) -> Path:
    out_dir = results_dir(run_id)
    write_json(out_dir / "provenance.json", system_provenance(extra=extra))
    freeze = pip_freeze()
    if freeze:
        (out_dir / "requirements.lock").write_text("\n".join(freeze) + "\n")
    return out_dir
