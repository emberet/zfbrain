"""Unpack a Claude-Design `.dc.html` bundle into plain static files under site/.

The bundle is one HTML file carrying a `__bundler/manifest` (gzip+base64
blobs: React UMD, the dc-runtime, fonts) and a `__bundler/template` (the real
page as a JSON string). Its loader turns those into blob: URLs at runtime.
This script does the same thing once, on disk, so the page becomes ordinary
source: index.html + styles/ + fonts/ + vendor/. The dc-runtime needs no
bundler (verified: zero references to __bundler / blob: / currentScript) and
skips its unpkg fetch when window.React / window.ReactDOM already exist, so
React is vendored and loaded first.

    python tools/unbundle.py "~/Downloads/ZFBRAIN Site.html" site

Run it once; after that, edit site/ directly (the design canvas is the
source of record for the *look*, site/ is the source of record for the code).
"""

import base64
import gzip
import json
import os
import re
import sys
from pathlib import Path

# Every bundle mints fresh uuids, so nothing here may be hardcoded: the
# vendor scripts are whatever the template loads by <script src="uuid">, and
# their names come from sniffing the decompressed bytes.
def _vendor_name(body, i):
    head = body[:400].decode("utf-8", "replace")
    if "dc-runtime" in head or "DCLogic" in head:
        return "vendor/dc-runtime.js"
    if "@ds-bundle" in head or "__ds_ns" in head:
        return "vendor/ds-bundle.js"
    return f"vendor/bundled-{i}.js"

FONT_FACE_RE = re.compile(
    r"/\*\s*(?P<subset>[\w-]+)\s*\*/\s*@font-face\s*\{(?P<body>.*?)\}", re.S)


def _block(html, kind):
    m = re.search(r'<script type="__bundler/%s"[^>]*>(.*?)</script>' % re.escape(kind), html, re.S)
    if not m:
        raise SystemExit(f"no __bundler/{kind} block in bundle")
    return m.group(1)


def _decode(entry):
    raw = base64.b64decode(entry["data"])
    return gzip.decompress(raw) if entry.get("compressed") else raw


def main(bundle, out):
    bundle = Path(os.path.expanduser(bundle))
    out = Path(out)
    html = bundle.read_text(encoding="utf-8")

    manifest = json.loads(_block(html, "manifest").strip())
    ext = json.loads(_block(html, "ext_resources").strip())
    # the template is a JSON string literal followed by trailing text
    template, _ = json.JSONDecoder().raw_decode(_block(html, "template").lstrip())

    for d in ("vendor", "fonts", "styles"):
        (out / d).mkdir(parents=True, exist_ok=True)

    names = {}
    # scripts the template pulls in by uuid
    for i, uuid in enumerate(re.findall(r'<script[^>]*src=\\?"([0-9a-f-]{36})', template)):
        names[uuid] = _vendor_name(_decode(manifest[uuid]), i)
    # externals (React and friends) keep their own filenames
    for r in ext:
        names[r["uuid"]] = "vendor/" + r["id"].rsplit("/", 1)[-1]

    # --- styles: two <style> blocks inside <helmet> -----------------------
    styles = re.findall(r"<style[^>]*>(.*?)</style>", template, re.S)
    if not styles:
        raise SystemExit("no <style> block in the template")
    # the first block is the design system (it carries the @font-face rules);
    # anything after it is page CSS
    industry, rest = styles[0], styles[1:]

    # fonts: name by family/weight/subset from every @font-face block in the
    # template (a design can carry more than one stylesheet)
    for m in FONT_FACE_RE.finditer(template):
        body, subset = m.group("body"), m.group("subset")
        fam = re.search(r"font-family:\s*'([^']+)'", body).group(1)
        weight = re.search(r"font-weight:\s*(\d+)", body).group(1)
        uuid = re.search(r'url\("([0-9a-f-]{36})"\)', body).group(1)
        fname = f"fonts/{fam.lower().replace(' ', '-')}-{weight}-{subset}.woff2"
        names[uuid] = fname

    for uuid, rel in names.items():
        if uuid not in manifest:
            raise SystemExit(f"manifest is missing {uuid} ({rel})")
        (out / rel).write_bytes(_decode(manifest[uuid]))
    unused = set(manifest) - set(names)
    if unused:
        raise SystemExit(f"unmapped manifest entries: {sorted(unused)}")

    def fix_fonts(css):
        for uuid, rel in names.items():
            if rel.startswith("fonts/"):
                css = css.replace(f'url("{uuid}")', f'url("../{rel}")')
        return css
    (out / "styles/industry.css").write_text(fix_fonts(industry).strip() + "\n", encoding="utf-8")
    (out / "styles/zf.css").write_text(
        "\n\n".join(fix_fonts(b).strip() for b in rest) + "\n", encoding="utf-8")

    # --- page --------------------------------------------------------------
    body = re.search(r"<body[^>]*>(.*)</body>", template, re.S).group(1)
    body = re.sub(r"<helmet>.*?</helmet>\s*", "", body, count=1, flags=re.S)
    for uuid, rel in names.items():
        body = body.replace(f'src="{uuid}"', f'src="{rel}"')
    body = body.strip()

    # React first (the runtime skips its own CDN fetch when window.React exists),
    # then the design-system bundle, then the runtime that boots the page
    order = ["react.production.min.js", "react-dom.production.min.js",
             "ds-bundle.js", "dc-runtime.js"]
    vend = [v for v in names.values() if v.startswith("vendor/")]
    vend.sort(key=lambda v: (order.index(v.split("/")[1]) if v.split("/")[1] in order else 99, v))
    scripts = "\n".join(f'<script src="{v}"></script>' for v in vend)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZFBRAIN</title>
<meta name="description" content="A living larval-zebrafish connectome roaming the open internet, funded with its own memecoin. Live telemetry, honest about what is and isn't real.">
<link rel="stylesheet" href="styles/industry.css">
<link rel="stylesheet" href="styles/zf.css">
{scripts}
</head>
<body>
{body}
</body>
</html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")

    total = sum((out / rel).stat().st_size for rel in names.values())
    print(f"wrote {out}/index.html ({len(page):,} B) + {len(names)} assets ({total:,} B)")
    for rel in sorted(names.values()):
        print(f"  {rel}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
