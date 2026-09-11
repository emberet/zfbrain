"""Post engine for the narrator. Caps posts per day, refuses duplicates.
Without POST_URL set it only prints, so the pipeline is safe to run empty.

The account is ZF_X_HANDLE (@zfbraindev). Use a handle you own: posting as
someone else's is impersonation, which is the one thing this project cannot
do and stay what it says it is.

Wire a real posting endpoint (X via a relay/tool of your choice) and set
POST_URL in .env. The journal always records what was accepted.
"""

import json
import os
import time
from pathlib import Path

import requests

STATE_DIR = Path(os.environ.get("ZF_STATE_DIR", ".state"))
SEEN = STATE_DIR / "posted.jsonl"
DAILY_CAP = int(os.environ.get("ZF_VOICE_DAILY_CAP", "8"))
HANDLE = os.environ.get("ZF_X_HANDLE", "zfbraindev")


def _seen():
    texts = []
    if SEEN.exists():
        for line in SEEN.read_text().splitlines():
            try:
                texts.append(json.loads(line)["text"])
            except Exception:  # noqa: BLE001
                pass
    return texts


def _today_count():
    if SEEN.exists():
        t0 = time.time() - 24 * 3600
        return sum(1 for line in SEEN.read_text().splitlines()
                   if json.loads(line)["t"] > t0)
    return 0


def post(text, force=False):
    if text in _seen():
        print("duplicate, skipped")
        return False
    if not force and _today_count() >= DAILY_CAP:
        print("daily cap reached")
        return False
    url = os.environ.get("POST_URL")
    if url:
        try:
            r = requests.post(url, json={"text": text, "handle": HANDLE}, timeout=30)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"post failed: {exc}")
            return False
    else:
        print(f"[would post as @{HANDLE}] {text}")
    with open(SEEN, "a") as f:
        f.write(json.dumps({"t": time.time(), "text": text}) + "\n")
    return True