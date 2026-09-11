"""The fish, loose on the open internet.

A headless Chromium is opened (only when ZF_ALLOW_BROWSER=1). A screenshot of
the live page is sampled through the retina, the whole connectome integrates,
and the fish's descending-motor populations drive the cursor, scroll bursts,
reversals and the Mauthner escape. There is no start and no stop - it roams
while the process is up, and if a run dies it waits and starts another life.

Rails (same reasoning as the fly project, stated plainly):
  * no wallet, no keyboard, no downloads
  * every click is checked before it lands and vetoed if it reads as
    a submit / sign-in / purchase / connect
  * a domain allowlist: FLY_ROAM_OPEN=1 removes it and should not be left on
  * a hop budget so a dead end does not become a permanent home

Usage:
    ZF_ALLOW_BROWSER=1 python roam.py            # http://localhost:4660/state
    python fishsim.py --smoke                    # brain without a browser
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from time import sleep

import numpy as np

from fishsim import FishSim, load_graph
from retina import Retina

DEFAULT_ALLOWLIST = [
    "wikipedia.org", "wikimedia.org", "wikisource.org", "gutenberg.org",
    "openlibrary.org", "arxiv.org", "xkcd.com", "ponsfamily.com", "blockscout.com",
]
HOME = "https://en.wikipedia.org/wiki/Zebrafish"
VETO_WORDS = (
    "submit", "sign in", "sign up", "login", "log in", "connect wallet",
    "buy", "pay", "purchase", "checkout", "install", "download", "upload",
    "launch token", "confirm", "join", "subscribe", "get started",
)


def _env_flag(name):
    return os.environ.get(name, "0").strip() in ("1", "true", "yes")


class Roamer:
    def __init__(self, graph_path, groups_path, state_dir=".state"):
        self.graph = load_graph(graph_path, groups_path)
        self.sim = FishSim(self.graph)
        self.retina = Retina()
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        # map retina + DSGC groups onto the connectome's own groups
        self.groups = self.graph["groups"]
        self._require_groups()
        self.stats = {"pages": 0, "clicks": 0, "scrolls": 0, "vetoed": 0,
                      "escapes": 0, "brain_steps": 0, "page": None}
        self._quiet_turns = 0
        self._life = 0
        self._last_detail = None

    def _require_groups(self):
        need = ["retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right",
                "nmlf", "vspn", "mauthner", "spinal"]
        missing = [g for g in need if not self.groups.get(g)]
        if missing:
            raise SystemExit(
                f"connectome is missing readout groups {missing}.\n"
                f"Wire them in data/raw/groups.csv (group,id), then rebuild.")
        self.retina_taken = {g: self.groups[g] for g in ("retina",)}
        self.motion_groups = {g: self.groups[g] for g in
                              ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right")}

    # ---- the veto ----------------------------------------------------
    def veto(self, page):
        try:
            el = page.evaluate(
                "() => { const e = document.elementFromPoint("
                "   window.innerWidth/2, window.innerHeight/2);"
                "   if (!e) return '';"
                "   const t = e.closest('button,[role=button],a,input,textarea,form');"
                "   return t ? ((t.innerText||'')+' '+(t.getAttribute('placeholder')||'')"
                "             +' '+(t.getAttribute('aria-label')||'')).toLowerCase() : ''; }")
        except Exception:
            el = ""
        for w in VETO_WORDS:
            if w in el:
                return w
        return None

    # ---- one life ----------------------------------------------------
    def _new_life(self, browser):
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=False,
        )
        page = ctx.new_page()
        page.goto(HOME, wait_until="domcontentloaded")
        self._life += 1
        self.stats["page"] = page.url
        self.stats["pages"] += 1
        return page

    def step(self, page, hops_left):
        frame = np.asarray(page.screenshot())
        out = self.retina.step(frame)
        rates = self.retina.rates(out, motion_gain=3.0)
        for group, hz in rates.items():
            self.sim.set_drive(self.groups.get(group, []), hz)
        detail = self.sim.run(400)
        self._last_detail = detail
        dec = self.sim.decode(detail["rates_hz"])
        self.stats["brain_steps"] += 400

        # steering -> cursor
        page.mouse.move(640 + int(dec["dx"] * 60), 400 + int(dec["dy"] * 60))
        # nMLF bout -> scroll burst
        if dec["scroll"] > 0.45:
            page.mouse.wheel(0, int(dec["scroll"] * 900))
            self.stats["scrolls"] += 1
        # vSPN reversal -> flick the other way
        if dec["turn"] > 0.5:
            page.mouse.wheel(0, -int(dec["turn"] * 400))

        # Mauthner escape -> dart away, veto everything pending
        if dec["escape"]:
            page.mouse.move(0 if np.random.rand() < 0.5 else 1280, 0)
            self.stats["escapes"] += 1
            return None, hops_left

        # quiet-freeze strike (the fish's answer to the fly's stop->click)
        strongest = max(dec["dx"], dec["dy"], key=abs)
        if abs(strongest) < 0.08:
            self._quiet_turns += 1
        else:
            self._quiet_turns = 0
        if self._quiet_turns >= 6:
            veto_reason = self.veto(page)
            if veto_reason:
                self.stats["vetoed"] += 1
            else:
                page.mouse.click(640, 400)
                self.stats["clicks"] += 1
                sleep(1.2)
                self._quiet_turns = 0
                hops_left -= 1

        return detail, hops_left

    def run(self, hops=None):
        if not _env_flag("ZF_ALLOW_BROWSER"):
            raise SystemExit(
                "A button in a config is not enough to open a browser against "
                "real sites. Set ZF_ALLOW_BROWSER=1 in .env first.")
        from playwright.sync_api import sync_playwright

        open_fence = _env_flag("ZF_ROAM_OPEN")
        allow = os.environ.get("ZF_ALLOWLIST", ",".join(DEFAULT_ALLOWLIST)).split(",")
        hops = hops or int(os.environ.get("ZF_ROAM_HOPS", "12"))

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            while True:
                page = self._new_life(browser)
                h = hops
                try:
                    while h > 0:
                        detail, h = self.step(page, h)
                        if not open_fence:
                            host = page.url.split("/")[2].lower()
                            if not any(d in host for d in allow):
                                page.goto(HOME, wait_until="domcontentloaded")
                                self.stats["page"] = page.url
                                self.stats["pages"] += 1
                        if detail is not None and np.random.rand() < 0.03:
                            self.heartbeat()
                        sleep(0.15)
                except Exception as exc:  # noqa: BLE001
                    print(f"life {self._life} died: {exc}")
                page.close()
                sleep(6)  # if a run dies, wait six seconds and start another life

    def heartbeat(self):
        brain = self._last_detail
        if brain is None:
            brain = self.sim.stream(0.0005, 400)
        state = {"status": "live",
                 "stats": self.stats,
                 "brain": brain,
                 "page": self.stats.get("page")}
        with open(os.path.join(self.state_dir, "live.json"), "w") as f:
            json.dump(state, f)
        site = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "site", "web", "live.json")
        try:
            with open(site, "w") as f:
                json.dump(state, f)
        except OSError:
            pass
        return state


def _pub(roamer):
    server = HTTPServer(("0.0.0.0", int(os.environ.get("ZF_ROAM_PORT", "4660"))),
                        make_handler(roamer))
    server.serve_forever()


def make_handler(roamer):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/state"):
                body = json.dumps(roamer.heartbeat()).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):  # keep the stream quiet
            pass
    return H


if __name__ == "__main__":
    import sys
    roamer = Roamer("build/graph.npz", "build/groups.json")
    Thread(target=_pub, args=(roamer,), daemon=True).start()
    try:
        roamer.run()
    except SystemExit as e:
        print(e)
        sys.exit(1)