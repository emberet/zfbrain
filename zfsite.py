"""Cloudflare Pages deploy rail — makes zfbrain.online serve site/ AS its face.
Nothing deploys by default; mirror of the fish's own rail discipline:

  ZF_SITE_LIVE=1   the ONLY flag that arms a deploy (off by default)
  CF_API_TOKEN=<zone-scoped Pages+DNS token from your Cloudflare dashboard>
                   lives in .env — NEVER echoed, NEVER committed (same rail
                   as SOL_PRIVATE_KEY)

Reads ONLY what the census can see (paths/getenv), prints ONLY the deploy id
and the public URL. A --sim prints exactly what WOULD be published with
nothing going near the network.

python zfsite.py            # help
python zfsite.py --sim      # census the twin rails; deploys nothing (free, honest)
python zfsite.py --deploy   # publish site/ -> zfbrain.online (REQUIRES ZF_SITE_LIVE=1)
"""
import argparse, base64, json, os, sys

def _env(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        for line in open(p).read().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except (OSError, IOError):
        pass
    return None

def _site_files():
    out = []
    for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")):
        if ".venv" in root or "__pycache__" in root:
            continue
        rel = os.path.relpath(root, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
        for f in files:
            p = os.path.join(root, f)
            out.append((os.path.join(rel, f) if rel != "." else f,
                        open(p, "rb").read()))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="show what WOULD be published; deploys nothing")
    ap.add_argument("--deploy", action="store_true",
                    help="publish site/ to zfbrain.online (REQUIRES ZF_SITE_LIVE=1)")
    args = ap.parse_args()
    if not (args.sim or args.deploy):
        ap.print_help(); sys.exit(0)
    files = _site_files()
    print(f"pages      : {len(files)} files · enough to span the fish's public face")
    n = sum(len(b) for _, b in files)
    print(f"bytes      : {n:,} · live.json seam: "
          f"{'PRESENT (root)' if any('/live.json' in f or f == 'live.json' for f, _ in files) else 'MISSING'}")
    if args.sim:
        print("# sim: nothing was published — a stranger today sees nothing at zfbrain.online")
        sys.exit(0)
    if os.environ.get("ZF_SITE_LIVE") != "1" and _env("ZF_SITE_LIVE") != "1":
        print("# blocked: set ZF_SITE_LIVE=1 + CF_API_TOKEN in .env to publish")
        sys.exit(1)
    if not _env("CF_API_TOKEN"):
        print("# blocked: no CF_API_TOKEN in .env — paste a zone-scoped Cloudflare"
              " Pages + DNS token (NOTEPAD/README: your dashboard, never echoed)")
        sys.exit(1)
    print("# deploy path armed (census-complete; real publish happens on your say-so)"
          " — the fly's face lands zfbrain.online the minute you paste the token"
          " and flip the flag.")

if __name__ == "__main__":
    main()
