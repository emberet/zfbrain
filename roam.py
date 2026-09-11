"""The fish, loose on the open internet — and the live feed the site shows.

A headless Chromium is opened (only when ZF_ALLOW_BROWSER=1). Every step: a
screenshot of the live page goes through the retina, the whole connectome
integrates 400 LIF steps, and the descending-motor populations drive the
cursor, scroll bouts, turns and the Mauthner escape. There is no start and no
stop — it roams while the process is up, and if a life dies it waits six
seconds and starts another.

Everything the fish just did is published, in-process, on 127.0.0.1:4660:

    GET /state      the heartbeat (JSON, see heartbeat() for the shape)
    GET /frame.jpg  what the fish is looking at right now (640x400 JPEG)
    GET /events     the same heartbeat as a Server-Sent-Events stream (~2 Hz)
    GET /graph      the running graph: counts, populations, where the layout is
    GET /graph.bin  every neuron's position + population, binary (once per page load)
    GET /firing.bin?seq=N  which neurons fired in the last 5 ms, one bit each
    GET /healthz    200 while the process is up

bin/tunnel.sh puts that behind https://live.zfbrain.online; the site
subscribes to it and shows the honest "asleep" state when nothing answers.
The server only ever reads the latest snapshot — every Playwright call stays
on the main thread.

Rails (same reasoning as the fly project, stated plainly):
  * no wallet, no keyboard, no downloads
  * every click is checked before it lands and vetoed if it reads as
    a submit / sign-in / purchase / connect
  * a domain allowlist: ZF_ROAM_OPEN=1 removes it and should not be left on
    (with it on, screenshots of whatever it wanders into are published)
  * a hop budget so a dead end does not become a permanent home

Usage:
    ZF_ALLOW_BROWSER=1 python roam.py            # or bin/roam.sh (sources .env)
    python fishsim.py                            # brain without a browser
"""

import io
import json
import os
import random
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

import solrpc
from fishsim import FishSim, load_graph
from retina import Retina

DEFAULT_ALLOWLIST = [
    "wikipedia.org", "wikimedia.org", "wikisource.org", "gutenberg.org",
    "openlibrary.org", "arxiv.org", "xkcd.com",
    "pump.fun", "raydium.io", "solscan.io", "explorer.solana.com",
]
HOME_SEEDS = [
    # the fish starts a fresh life at a random one of these, then wanders.
    # no captcha-walled sites here (solscan etc.) — they freeze the browser.
    "https://en.wikipedia.org/wiki/Zebrafish",
    "https://en.wikipedia.org/wiki/Fish",
    "https://www.gutenberg.org/",
    "https://openlibrary.org/",
    "https://arxiv.org/list/q-bio.NC/recent",
    "https://xkcd.com/",
    "https://en.wikipedia.org/wiki/Aquarium",
    "https://archive.org/",
    "https://en.wikisource.org/",
]
HOME = HOME_SEEDS[0]  # fallback only; _new_life picks a random seed
VETO_WORDS = (
    "submit", "sign in", "sign up", "login", "log in", "connect wallet",
    "buy", "pay", "purchase", "checkout", "install", "download", "upload",
    "launch token", "confirm", "join", "subscribe", "get started",
    "captcha", "verify you are human", "i'm not a robot", "challenge",
)
VIEW_W, VIEW_H = 1280, 800        # the fish's viewport
FRAME_W, FRAME_H = 640, 400       # the published frame
SIM_STEPS = 400                   # 400 x 0.5 ms = 0.2 s of biological time per step
DEFAULT_ORIGINS = ("https://zfbrain.online,https://www.zfbrain.online,"
                   "https://zfbrain.pages.dev,http://localhost:8787")


def _env_flag(name):
    return os.environ.get(name, "0").strip() in ("1", "true", "yes")


def _short_url(url):
    if not url:
        return None
    u = urlsplit(url)
    return (u.netloc + u.path).rstrip("/")


# ---------------------------------------------------------------------------
# the feed: the latest snapshot, shared between the fish and the HTTP threads
# ---------------------------------------------------------------------------
class Feed:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.state = {"v": 2, "status": "starting", "seq": 0, "ts": time.time()}
        self.frame = b""
        self.frame_seq = 0
        self.mask = b""          # packed firing bits, one per neuron
        self.mask_seq = 0
        self.sse_clients = 0

    def publish(self, state, frame=None, mask=None):
        with self.cond:
            self.state = state
            if frame is not None:
                self.frame = frame
                self.frame_seq = state["frame"]["seq"]
            if mask is not None:
                self.mask = mask
                self.mask_seq = state["brain"]["firing_seq"]
            self.cond.notify_all()

    def wait_for_new(self, seen_seq, timeout):
        """Block until a snapshot newer than seen_seq exists (or timeout)."""
        with self.cond:
            self.cond.wait_for(lambda: self.state["seq"] != seen_seq, timeout)
            return self.state

    def snapshot(self):
        with self.lock:
            return self.state, self.frame, self.frame_seq

    def mask_snapshot(self):
        with self.lock:
            return self.mask, self.mask_seq


class Roamer:
    def __init__(self, graph_path, groups_path, state_dir=".state"):
        self.graph = load_graph(graph_path, groups_path)
        self.meta = self.graph["meta"]
        self.sim = FishSim(self.graph)
        self.retina = Retina()
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.groups = self.graph["groups"]
        self._require_groups()

        self.stats = {"pages": 0, "clicks": 0, "scrolls": 0, "vetoed": 0,
                      "escapes": 0, "brain_steps": 0}
        self.events = deque(maxlen=12)
        self.feed = Feed()
        self._seq = 0
        self._mask_seq = 0
        self._mask_at = 0.0
        self._life = 0
        self._seed = HOME_SEEDS[0]
        self._quiet_turns = 0
        self._cursor = (VIEW_W // 2, VIEW_H // 2)
        self._page_url = None
        self._page_title = None
        self._reafference = 0      # steps left in which the fish's own bout can't startle it
        self._bout_cool = 0        # a bout is followed by a glide, not another bout
        self._rest = {}            # running resting rates: a bout is a rise above them
        self._last_escape = 0.0
        self._last_file_write = 0.0
        self._chain = {"network": "mainnet", "slot": None, "at": 0.0}
        self.graph_doc = self._graph_doc()
        threading.Thread(target=self._chain_loop, daemon=True).start()

    def _graph_doc(self):
        """What the site draws: every neuron's position and population. Built
        once. /graph is the small JSON description; /graph.bin is the layout —
        float32 (x, y) per neuron in the fish frame (x 0..1 head->tail, y
        up-negative) followed by a uint16 population id per neuron. The
        synthetic graph is already in that frame; a real EM volume gets
        projected to the lateral view here, so the page never changes."""
        names = sorted(self.groups)
        gid = np.full(self.sim.n, 65535, dtype=np.uint16)
        for i, name in enumerate(names):
            gid[np.asarray(self.groups[name], dtype=np.int64)] = i
        xy = np.asarray(self.sim.coords, dtype=np.float32)[:, :2].copy()
        if self.meta.get("source") not in ("synthetic", "smoke"):
            # lateral projection of real somata into the fish frame: x along
            # the body, y dorso-ventral, both normalised to the drawn silhouette
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            span = np.maximum(hi - lo, 1e-6)
            xy[:, 0] = (xy[:, 0] - lo[0]) / span[0]
            xy[:, 1] = (xy[:, 1] - lo[1]) / span[1] * 0.26 - 0.13
        self.graph_bin = xy.astype("<f4").tobytes() + gid.astype("<u2").tobytes()
        doc = {"n": int(self.sim.n), "label": self.meta["label"], "source": self.meta.get("source"),
               "built_at": self.meta.get("built_at"), "synapses": self.meta.get("synapses"),
               "groups": names, "layout": "/graph.bin", "layout_bytes": len(self.graph_bin)}
        return json.dumps(doc, separators=(",", ":")).encode()

    def _require_groups(self):
        need = ["retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right",
                "nmlf", "vspn", "mauthner", "spinal"]
        missing = [g for g in need if not len(self.groups.get(g, []))]
        if missing:
            raise SystemExit(
                f"connectome is missing readout groups {missing}.\n"
                f"Wire them in data/raw/groups.csv (group,id), then rebuild.")

    # ---- events + chain -------------------------------------------------
    def _event(self, kind, msg):
        self.events.append({"t": round(time.time(), 2), "kind": kind, "msg": msg})

    def _chain_loop(self):
        """One honest chain number for the header: the current mainnet slot."""
        while True:
            try:
                self._chain = {"network": "mainnet",
                               "slot": solrpc.get_slot(url=solrpc.MAINNET),
                               "at": time.time()}
            except Exception:  # noqa: BLE001 — keep the last value, age grows
                pass
            time.sleep(10)

    # ---- the veto ----------------------------------------------------
    def veto(self, page, x, y):
        try:
            el = page.evaluate(
                "([x, y]) => { const e = document.elementFromPoint(x, y);"
                "   if (!e) return '';"
                "   const t = e.closest('button,[role=button],a,input,textarea,form');"
                "   return t ? ((t.innerText||'')+' '+(t.getAttribute('placeholder')||'')"
                "             +' '+(t.getAttribute('aria-label')||'')).toLowerCase() : ''; }",
                [x, y])
        except Exception:  # noqa: BLE001
            el = ""
        for w in VETO_WORDS:
            if w in el:
                return w
        return None

    # ---- escape: dart to a fresh seed (a startled larva leaves the page) ----
    def _look(self, page, cx, cy, max_dist=180):
        """A bored fish looks for a link it can actually follow. Returns True
        if it found a safe one within reach and clicked it, else False.
        Scanning is greedy from the point of gaze outward, veto-clean.
        Links that stay on the same URL (SPA dead ends) and file downloads
        (which the context refuses) are skipped -- a click must change the
        page or it was never a real option."""
        before = page.url
        try:
            near = page.evaluate(
                """([cx, cy, r, here]) => {
                     const els = Array.from(document.querySelectorAll('a[href]'));
                     const scored = [];
                     for (const el of els) {
                       const href = (el.getAttribute('href') || '').trim();
                       if (!href || href.startsWith('#') || href.includes('javascript:')) continue;
                       const url = new URL(href, here);
                       if (url.href === here) continue;          // dead end: same page
                       if (url.protocol !== 'http:' && url.protocol !== 'https:') continue;
                       if (/\\.(pdf|zip|tar|gz|exe|dmg|iso|tar\\.gz)(\\?|$)/i.test(url.pathname)) continue;
                       const t = (el.innerText || '').trim();
                       if (!t || t.length > 120) continue;
                       const b = el.getBoundingClientRect();
                       if (b.width === 0 && b.height === 0) continue;
                       const x = b.left + b.width / 2, y = b.top + b.height / 2;
                       const d = Math.hypot(x - cx, y - cy);
                       if (d <= r) scored.push({x, y, d, t});
                     }
                     scored.sort((a, b) => a.d - b.d);
                     return scored; }""",
                [cx, cy, max_dist, before])
        except Exception:  # noqa: BLE001
            return False
        for link in near:
            x, y = int(link["x"]), int(link["y"])
            if self.veto(page, x, y):
                continue
            try:
                page.mouse.move(x, y)
                page.mouse.click(x, y)
            except Exception:  # noqa: BLE001 — a download/JS link must not kill the life
                continue
            self.stats["clicks"] += 1
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:  # noqa: BLE001
                pass
            if page.url == before:  # dead click — try the next candidate
                continue
            self._event("look", "wants something new · followed a link")
            time.sleep(0.8)
            return True
        if self.veto(page, cx, cy):
            self.stats["vetoed"] += 1
            self._event("veto", "wanted a link, all in reach were unsafe")
        else:
            self._event("look", "no real links in reach · tried them all")
        return False

    def _is_captcha_page(self, page):
        try:
            title = page.title().lower()
        except Exception:  # noqa: BLE001
            title = ""
        if any(w in title for w in ("captcha", "verify you are human",
                                    "i'm not a robot", "challenge", "cf-error")):
            return True
        try:
            txt = page.evaluate(
                "() => (document.body ? document.body.innerText : '').toLowerCase()")
        except Exception:  # noqa: BLE001
            return False
        return any(w in txt for w in ("cf-challenge", "enable javascript and cookies",
                                      "checking your browser before accessing",
                                      "verify you are human", "i'm not a robot",
                                      "press and hold the button",
                                      "captcha required"))
    def _escape_to_fresh(self, page):
        others = [h for h in HOME_SEEDS if h != self._seed]
        seed = random.choice(others) if others else HOME_SEEDS[0]
        self._seed = seed
        self._event("escape", f"Mauthner fired · darted to {_short_url(seed)}")
        try:
            page.goto(seed, wait_until="domcontentloaded")
        except Exception:  # noqa: BLE001 — a dead seed must not crash the life
            self._seed = HOME_SEEDS[0]
            page.goto(self._seed, wait_until="domcontentloaded")
        self._page_url = None

    # ---- one life ----------------------------------------------------
    def _new_life(self, browser):
        ctx = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H},
            accept_downloads=False,
        )
        page = ctx.new_page()
        page.set_default_timeout(15000)
        page.on("popup", lambda p: p.close())  # no popups, one page per life
        seed = random.choice(HOME_SEEDS)
        self._seed = seed
        try:
            page.goto(seed, wait_until="domcontentloaded")
        except Exception:  # noqa: BLE001 — a dead seed must not end the life
            seed = HOME_SEEDS[0]
            self._seed = seed
            page.goto(seed, wait_until="domcontentloaded")
        self._life += 1
        self._quiet_turns = 0
        self._cursor = (VIEW_W // 2, VIEW_H // 2)
        self._page_url = None
        self.stats["pages"] += 1
        self._event("life", f"life {self._life} started · {_short_url(seed)}")
        return page

    def _note_page(self, page):
        url = page.url
        if url != self._page_url:
            if self._page_url is not None:
                self.stats["pages"] += 1
                self._event("page", f"page → {_short_url(url)}")
            self._page_url = url
            try:
                self._page_title = page.title()
            except Exception:  # noqa: BLE001
                self._page_title = None

    def step(self, page, hops_left):
        # what the fish sees: one screenshot feeds the retina AND the site
        jpg = page.screenshot(type="jpeg", quality=70)
        img = Image.open(io.BytesIO(jpg)).convert("RGB")
        out = self.retina.step(np.asarray(img))
        rates = self.retina.rates(out, motion_gain=1.5)
        for group, hz in rates.items():
            self.sim.set_drive(self.groups.get(group, []), hz)
        detail = self.sim.run(SIM_STEPS)
        for g in ("nmlf", "vspn", "mauthner"):  # slow baselines (~8 s); a burst rides above them
            hz = detail["rates_hz"].get(g, 0.0)
            self._rest[g] = hz if g not in self._rest else 0.95 * self._rest[g] + 0.05 * hz
        dec = self.sim.decode(detail["rates_hz"], rest=self._rest)
        self.stats["brain_steps"] += SIM_STEPS
        self._note_page(page)

        # steering -> the cursor drifts with the DSGC asymmetry, like a heading,
        # with a slow pull back to the middle of the field
        ox, oy = self._cursor
        cx = int(min(VIEW_W - 1, max(0, ox + dec["dx"] * 200 - (ox - VIEW_W / 2) * 0.02)))
        cy = int(min(VIEW_H - 1, max(0, oy + dec["dy"] * 200 - (oy - VIEW_H / 2) * 0.02)))
        page.mouse.move(cx, cy)
        self._cursor = (cx, cy)
        # nMLF bout -> scroll burst
        moved = False
        if self._bout_cool:
            self._bout_cool -= 1
        elif dec["scroll"] > 0.3:
            px = 240 + int(dec["scroll"] * 480)
            page.mouse.wheel(0, px)
            self.stats["scrolls"] += 1
            self._event("bout", f"nMLF bout · scroll down {px} px")
            self._bout_cool = 4  # ~1.5 s glide
            moved = True
        # vSPN reversal -> flick the other way
        if dec["turn"] > 0.5 and not self._bout_cool:
            px = int(dec["turn"] * 300)
            page.mouse.wheel(0, -px)
            self._event("turn", f"vSPN turn · scroll up {px} px")
            moved = True
        # corollary discharge: the whole-field motion a bout makes is the fish's
        # own doing and must not read as a looming object for the next steps
        if moved:
            self._reafference = 3
        elif self._reafference:
            self._reafference -= 1

        # Mauthner escape -> dart away: a startled larva clears the field and
        # lands on a fresh seed elsewhere; it never pins itself to the same page.
        if dec["escape"] and not self._reafference and time.time() - self._last_escape > 3.0:
            self._last_escape = time.time()
            self.stats["escapes"] += 1
            page.mouse.move(0 if np.random.rand() < 0.5 else VIEW_W - 1, 0)
            self._quiet_turns = 0
            self._escape_to_fresh(page)
            self._publish(detail, dec, out, img)
            return detail, hops_left

        # quiet-freeze strike: a fish that has been still for a while strikes.
        # "what the fish is looking at" = the page; a strike means it wants
        # something new, so aim at a real link near the cursor. No link within
        # reach (text, whitespace, a dead end) = this page is exhausted →
        # wander to a fresh seed rather than click dead pixels.
        quiet = abs(dec["dx"]) < 0.08 and abs(dec["dy"]) < 0.08 and dec["scroll"] < 0.3 and not moved
        self._quiet_turns = self._quiet_turns + 1 if quiet else 0
        if self._quiet_turns >= 15:
            self._quiet_turns = 0
            if self._look(page, cx, cy):
                hops_left -= 1
            else:
                self._event("wander", "no links in reach · wandering to a fresh seed")
                self._escape_to_fresh(page)

        self._publish(detail, dec, out, img)
        return detail, hops_left

    # ---- the heartbeat ------------------------------------------------
    def heartbeat(self, detail=None, dec=None, out=None, status="live"):
        """Everything the site shows, in one dict. Pure: reads state, writes
        nothing. Numbers are the running graph's — never a placeholder."""
        self._seq += 1
        now = time.time()
        chain = self._chain
        state = {
            "v": 2, "status": status, "ts": round(now, 3), "seq": self._seq,
            "life": self._life,
            "graph": {"neurons": self.meta["neurons"], "synapses": self.meta["synapses"],
                      "label": self.meta["label"]},
            "page": {"url": self._page_url, "short": _short_url(self._page_url),
                     "title": self._page_title},
            "cursor": {"x": round(self._cursor[0] / VIEW_W, 4),
                       "y": round(self._cursor[1] / VIEW_H, 4)},
            "frame": {"seq": self._seq, "w": FRAME_W, "h": FRAME_H},
            "retina": None,
            "brain": None,
            "decode": dec,
            "stats": dict(self.stats),
            "events": list(self.events),
            "chain": {"network": chain["network"], "slot": chain["slot"],
                      "age_s": round(now - chain["at"]) if chain["slot"] else None},
        }
        if out is not None:
            lum = out["luminance"]
            state["retina"] = {
                "rows": int(lum.shape[0]), "cols": int(lum.shape[1]),
                "lum": [round(float(v), 3) for v in lum.reshape(-1)],
                "flow": {k: round(float(g.mean()) * 100, 2) for k, g in out["motion"].items()},
            }
        if detail is not None:
            state["brain"] = {
                "t": detail["t"],
                "spikes_per_sec": detail["spikes_per_sec"],
                "mean_membrane_mv": detail["mean_membrane_mv"],
                "firing": detail["firing_recent"],          # spiked in the last 5 ms
                "firing_seq": self._mask_seq,               # which /firing.bin these bits are
                "habituation": detail["habituation"],       # sensory synaptic resource used, 0-1
                "rates_hz": detail["rates_hz"],
            }
        return state

    def _publish(self, detail, dec, out, img):
        small = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=60, optimize=True)
        # the firing bits are 23 KB at whole-brain scale: publish at most 1 Hz,
        # and as bytes on their own route — the hero cannot show more anyway
        mask = None
        now = time.time()
        if detail is not None and now - self._mask_at >= 1.0:
            mask = detail["firing_bits"]
            self._mask_seq += 1
            self._mask_at = now
        state = self.heartbeat(detail, dec, out)
        self.feed.publish(state, buf.getvalue(), mask)
        if time.time() - self._last_file_write >= 1.0:
            self._write_state(state)

    def _write_state(self, state):
        base = os.path.dirname(os.path.abspath(__file__))
        seams = [os.path.join(self.state_dir, "live.json"),     # 1 brain (REAL)
                 os.path.join(base, "site", "live.json"),       # 2 ROOT legal open twin
                 os.path.join(base, "site", "web", "live.json")]  # 3 web twin
        try:
            for seam in seams:
                os.makedirs(os.path.dirname(seam), exist_ok=True)
                if seam.endswith("/site/live.json") or seam == base + "/site/live.json" or "/site/live.json" in seam:
                    _tmp = seam + ".core.tmp"
                else:
                    _tmp = seam + ".tmp"
                with open(_tmp, "w") as f:
                    json.dump(state, f)
                os.replace(_tmp, seam)
            self._last_file_write = time.time()
        except OSError as exc:
            print(f"live.json seam: {exc}", file=sys.stderr)

    # ---- the loop -----------------------------------------------------
    def run(self, hops=None):
        if not _env_flag("ZF_ALLOW_BROWSER"):
            raise SystemExit(
                "A button in a config is not enough to open a browser against "
                "real sites. Set ZF_ALLOW_BROWSER=1 in .env first.")
        from playwright.sync_api import sync_playwright

        open_fence = _env_flag("ZF_ROAM_OPEN")
        allow = [d.strip() for d in
                 os.environ.get("ZF_ALLOWLIST", ",".join(DEFAULT_ALLOWLIST)).split(",") if d.strip()]
        hops = hops or int(os.environ.get("ZF_ROAM_HOPS", "40"))
        if open_fence:
            print("ZF_ROAM_OPEN=1: the allowlist is OFF — screenshots of anything it "
                  "wanders into will be published", file=sys.stderr)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            while True:
                try:
                    page = self._new_life(browser)
                except Exception as exc:  # noqa: BLE001
                    print(f"life could not start: {exc}", file=sys.stderr)
                    time.sleep(6)
                    continue
                h = hops
                try:
                    while h > 0:
                        _, h = self.step(page, h)
                        host = urlsplit(page.url).netloc.lower()
                        if not host:  # about:blank and friends — nothing to see
                            self._event("fence", "blank page · back to seed")
                            page.goto(self._seed, wait_until="domcontentloaded")
                        elif not open_fence and not any(d in host for d in allow):
                            self._event("fence", f"{host} is off the allowlist · back to seed")
                            page.goto(self._seed, wait_until="domcontentloaded")
                        elif self._is_captcha_page(page):
                            self._event("fence", "captcha wall · dart away")
                            self._escape_to_fresh(page)
                        time.sleep(0.15)
                    self._event("life", f"life {self._life} · hop budget spent")
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).splitlines()[0][:120]
                    print(f"life {self._life} died: {msg}", file=sys.stderr)
                    self._event("life", f"life {self._life} died · {msg}")
                try:
                    page.context.close()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(6)  # if a run dies, wait six seconds and start another life


# ---------------------------------------------------------------------------
# the feed server
# ---------------------------------------------------------------------------
def make_handler(roamer):
    feed = roamer.feed
    origins = {o.strip() for o in
               os.environ.get("ZF_LIVE_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()}
    max_sse = int(os.environ.get("ZF_LIVE_MAX_SSE", "200"))

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send(self, code, body, ctype, cache="no-store", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            for k, v in extra:
                self.send_header(k, v)
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Cache-Control")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send(200, b"ok", "text/plain")
            elif path in ("/", "/state"):
                state, _, _ = feed.snapshot()
                self._send(200, json.dumps(state).encode(), "application/json",
                           cache="public, max-age=1")
            elif path == "/graph":
                self._send(200, roamer.graph_doc, "application/json",
                           cache="public, max-age=300",
                           extra=[("ETag", f'"{roamer.meta.get("built_at", "0")}"')])
            elif path == "/firing.bin":
                mask, seq = feed.mask_snapshot()
                if not mask:
                    self._send(404, b"no firing mask yet", "text/plain")
                    return
                cache = "public, max-age=60" if "?" in self.path else "no-store"
                self._send(200, mask, "application/octet-stream", cache=cache,
                           extra=[("ETag", f'"{seq}"')])
            elif path == "/graph.bin":
                self._send(200, roamer.graph_bin, "application/octet-stream",
                           cache="public, max-age=86400",
                           extra=[("ETag", f'"{roamer.meta.get("built_at", "0")}-bin"')])
            elif path == "/frame.jpg":
                _, frame, seq = feed.snapshot()
                if not frame:
                    self._send(404, b"no frame yet", "text/plain")
                    return
                cache = "public, max-age=60" if "?" in self.path else "no-store"
                self._send(200, frame, "image/jpeg", cache=cache,
                           extra=[("ETag", f'"{seq}"')])
            elif path == "/events":
                self._events()
            else:
                self._send(404, b"not found", "text/plain")

        def _events(self):
            with feed.lock:
                if feed.sse_clients >= max_sse:
                    busy = True
                else:
                    feed.sse_clients += 1
                    busy = False
            if busy:
                self._send(429, b"too many live viewers; poll /state", "text/plain",
                           extra=[("Retry-After", "5")])
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self._cors()
                self.end_headers()
                seen = -1
                while True:
                    t0 = time.time()
                    state = feed.wait_for_new(seen, timeout=15)
                    if state["seq"] == seen:
                        self.wfile.write(b": ping\n\n")
                    else:
                        seen = state["seq"]
                        self.wfile.write(b"data: " + json.dumps(state).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(max(0.0, 0.5 - (time.time() - t0)))  # <= 2 Hz per client
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with feed.lock:
                    feed.sse_clients -= 1

        def log_message(self, *a):  # keep the stream quiet
            pass

    return H


def serve(roamer):
    host = "127.0.0.1"  # the tunnel connects locally; nothing else should
    port = int(os.environ.get("ZF_ROAM_PORT", "4660"))
    server = ThreadingHTTPServer((host, port), make_handler(roamer))
    server.daemon_threads = True
    print(f"feed on http://{host}:{port}  (/state /frame.jpg /firing.bin /events /graph /healthz)")
    server.serve_forever()


if __name__ == "__main__":
    roamer = Roamer(os.environ.get("ZF_GRAPH", "build/graph.npz"), "build/groups.json",
                    os.environ.get("ZF_STATE_DIR", ".state"))
    threading.Thread(target=serve, args=(roamer,), daemon=True).start()
    try:
        roamer.run()
    except SystemExit as e:
        print(e, file=sys.stderr)
        sys.exit(1)
