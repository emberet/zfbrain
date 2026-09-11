"""The site rail — publishes site/ to zfbrain.online (Cloudflare Pages).

Nothing deploys by default; same discipline as the fish's other rails:

  ZF_SITE_LIVE=1     the ONLY flag that arms a deploy (off by default)
  CF_API_TOKEN       zone-scoped Cloudflare token in .env (Pages:Edit, DNS:Edit,
                     Cloudflare Tunnel:Edit) — NEVER echoed, NEVER committed
  CF_ACCOUNT_ID      the account that owns the zone (defaults to ours)

--sim  census: what WOULD be published, and whether the live feed answers.
--deploy  runs `wrangler pages deploy site` with an EMPTY env file so nothing
          from .env can ride along as a Pages secret, then prints the URL.

python zfsite.py            # help
python zfsite.py --sim      # census only; deploys nothing (free, honest)
python zfsite.py --deploy   # publish site/ -> zfbrain.online (REQUIRES ZF_SITE_LIVE=1)
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PROJECT = "zfbrain"
BRANCH = "main"
DEFAULT_ACCOUNT = "1b830f55dc609c3dd6f6fe2d30daee47"
LIVE = "https://live.zfbrain.online"
SKIP_DIRS = {".venv", "__pycache__", "node_modules", ".wrangler"}


def _env(name, default=None):
    v = os.environ.get(name)
    if v:
        return v
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].split("#", 1)[0].strip() or default
    except OSError:
        pass
    return default


def _site_files():
    out = []
    for root, dirs, files in os.walk(SITE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f == ".DS_Store":
                continue
            p = Path(root) / f
            out.append((str(p.relative_to(SITE)), p.stat().st_size))
    return sorted(out)


def census():
    files = _site_files()
    n = sum(b for _, b in files)
    print(f"pages      : {len(files)} files · {n:,} bytes")
    for rel, b in files:
        print(f"             {rel:48s} {b:>9,}")
    try:
        r = requests.get(LIVE + "/healthz", timeout=6)
        print(f"live feed  : {LIVE} -> {r.status_code} {'(the fish is up)' if r.ok else ''}")
    except requests.RequestException as exc:
        print(f"live feed  : {LIVE} unreachable ({type(exc).__name__}) — the page will say asleep")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="show what WOULD be published; deploys nothing")
    ap.add_argument("--deploy", action="store_true",
                    help="publish site/ to zfbrain.online (REQUIRES ZF_SITE_LIVE=1)")
    args = ap.parse_args()
    if not (args.sim or args.deploy):
        ap.print_help()
        sys.exit(0)

    census()
    if args.sim:
        print("# sim: nothing was published")
        sys.exit(0)

    if _env("ZF_SITE_LIVE") != "1":
        print("# blocked: set ZF_SITE_LIVE=1 (env or .env) to publish")
        sys.exit(1)
    token = _env("CF_API_TOKEN")
    if not token:
        print("# blocked: no CF_API_TOKEN in .env — paste a zone-scoped Cloudflare"
              " token (Pages:Edit + DNS:Edit); never echoed")
        sys.exit(1)

    env = dict(os.environ,
               CLOUDFLARE_API_TOKEN=token,
               CLOUDFLARE_ACCOUNT_ID=_env("CF_ACCOUNT_ID", DEFAULT_ACCOUNT),
               WRANGLER_SEND_METRICS="false")
    # an empty env file: wrangler must not read .env (the Solana seed lives there)
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as empty:
        empty_path = empty.name
    cmd = ["npx", "--yes", "wrangler", "pages", "deploy", str(SITE),
           "--project-name", PROJECT, "--branch", BRANCH, "--commit-dirty", "true",
           "--env-file", empty_path]
    print("# deploying:", " ".join(c for c in cmd if not c.startswith("/")))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    finally:
        os.unlink(empty_path)
    out = (proc.stdout + proc.stderr).replace(token, "<CF_API_TOKEN>")
    print("\n".join(line for line in out.splitlines() if line.strip())[-1200:])
    if proc.returncode != 0:
        print(f"# deploy failed (exit {proc.returncode})")
        sys.exit(proc.returncode)
    print("# published: https://zfbrain.online (and www) — read it there, not here")


if __name__ == "__main__":
    main()
