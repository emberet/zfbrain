"""Letting a stranger show the fish four words, and reading back what it said.

This is `talk.py` (commit 050b7d1) with the narrator cut out. talk.py was
reverted the same day it landed, for one recorded reason: it made an Anthropic
call per stranger greeting, a ceiling of about 4,300 a day. There is no model on
any path in this file. `line()` is a template, the fish's own words come from
lexicon.py's quantiser, and the whole feature runs with ANTHROPIC_API_KEY unset.

Two closed lists face each other and that is the entire moderation policy:

  * The stranger picks up to MAX_WORDS words **from data/basic_english.txt**.
    Not a regex over free text — membership in the same 850 words the fish has.
    Nothing a visitor types can be anything other than one of 850 known
    English nouns and verbs, so there is no injection surface and no slur
    surface, and the check is a set lookup rather than a blocklist someone has
    to keep maintaining.
  * The words are *drawn* onto a page-sized canvas, never rendered as HTML.
    There is no parser anywhere on this path.

The reply is the measured reaction of the same 187,053 neurons that read every
other page, plus the words lexicon.py's map puts on that reaction. Nothing here
touches the browser: no navigation, no click, no mouse. The message becomes
light on a retina and nothing else.

Budget: ZF_SHOW_PER_6H messages per visitor per 6 hours, persisted to
.state/show_budget.json so a restart cannot refill anyone's bucket. That is
fairness. Load is a separate concern and has separate controls — INBOX_MAX and
a minimum gap between greetings — so that the fish still spends its life
browsing. The page says plainly that it may not get to you.
"""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

VIEW_W, VIEW_H = 1280, 800     # the fish's viewport, same as roam.py's

MAX_WORDS = 4
INBOX_MAX = 8                  # waiting messages; over this, politely full
RING_MAX = 8                   # finished exchanges the site shows
GLOBAL_S = 20.0                # the fish gets at least this long between
                               # greetings, so browsing stays the default
WINDOW_S = 6 * 3600.0          # the budget window
PER_WINDOW = 2                 # ...and what fits in it, per visitor
BUDGET_PATH = Path(".state/show_budget.json")

# A page is mostly white and drives the retina in a regime the fish is used to.
# A dark card would read as a dim room and the brain would simply go quiet, so
# the greeting is drawn the way a page is: dark text on light.
BG = (232, 233, 238)
FG = (18, 20, 30)
ACCENT = (86, 130, 170)

FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)


def sanitise(text, words):
    """(clean, reason). `words` is the 850-word set; membership in it is the
    whole check. `reason` is None when the message is usable, otherwise a
    sentence the endpoint hands straight back to the visitor."""
    if not isinstance(text, str):
        return None, "a message has to be text"
    parts = text.lower().replace(",", " ").split()
    if not parts:
        return None, "nothing to show it"
    if len(parts) > MAX_WORDS:
        return None, (f"{MAX_WORDS} words at most — that is as long as its own "
                      f"reply is allowed to be")
    bad = [p for p in parts if p not in words]
    if bad:
        return None, (f"not in its word list: {' '.join(bad[:3])} — it knows "
                      f"{len(words)} words and those are all of them")
    return " ".join(parts), None


def _font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)    # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text, font, width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and draw.textlength(trial, font=font) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def render(text, size=64, pad=0):
    """The message as a page-sized image. Drawn, never HTML — there is no
    parser anywhere on this path, so there is nothing for a visitor string to
    be injected into. Shrinks the type until it fits rather than truncating.

    `pad` makes the canvas that much taller than the viewport, so the roamer
    can slide a viewport-sized window down it and give the retina real optic
    flow. A still image produces none after the first frame, and a fish with no
    flow is a fish with nothing to say."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (VIEW_W, VIEW_H + pad), BG)
    d = ImageDraw.Draw(img)
    margin = 120
    size = max(18, int(size))
    while True:
        font = _font(size)
        lines = _wrap(d, text, font, VIEW_W - 2 * margin)
        step = int(size * 1.35)
        if size <= 18 or len(lines) * step <= VIEW_H - 2 * margin:
            break
        size = max(18, int(size * 0.85))
    y = max(margin // 2, (VIEW_H + pad - len(lines) * step) // 2)
    for ln in lines:
        w = d.textlength(ln, font=font)
        d.text(((VIEW_W - w) / 2, y), ln, font=font, fill=FG)
        y += step
    d.rectangle([40, 40, VIEW_W - 40, VIEW_H + pad - 40], outline=ACCENT, width=3)
    return img


def line(reaction):
    """The reaction as one sentence, by template. Every number in it is one of
    the measured numbers — this function cannot say anything the run did not
    do, which is the whole reason it exists and the whole reason this feature
    has no API bill."""
    r = reaction.get("peak", {})
    mau, nmlf, vspn = r.get("mauthner", 0.0), r.get("nmlf", 0.0), r.get("vspn", 0.0)
    dsgc = max((r.get(f"dsgc_{k}", 0.0) for k in ("up", "down", "left", "right")),
               default=0.0)
    if reaction.get("escape"):
        head = f"Mauthner {mau:.1f} Hz — it startled and darted away from it."
    elif reaction.get("bout"):
        head = f"nMLF {nmlf:.1f} Hz — a swim bout, the way it moves down a page."
    elif reaction.get("turn"):
        head = f"vSPN {vspn:.1f} Hz — it turned aside."
    elif dsgc >= 1.0:
        head = f"DSGC {dsgc:.1f} Hz — its gaze drifted with the letters, no bout."
    else:
        head = "Nothing rose above its baseline. It held still."
    said = reaction.get("said") or []
    tail = (f" It said: {' '.join(said)}." if said else
            " Its map had no word for that state.")
    return (f"{head} {reaction.get('mean_hz', 0.0):.1f} Hz across the graph over "
            f"{reaction.get('thoughts', 0)} thoughts.{tail}")


class Show:
    """The inbox, the transcript ring, and the budget between them.

    One lock covers all of it. The roamer's sim thread is the only caller of
    `next_message`/`finish`; the HTTP threads call `submit`/`ring`. Everything
    that leaves goes out as a copy, so no caller can hold a reference into the
    ring."""

    def __init__(self, words, per_window=PER_WINDOW, window_s=WINDOW_S,
                 global_s=GLOBAL_S, budget_path=BUDGET_PATH):
        self.lock = threading.Lock()
        self.words = set(words)
        self.inbox = deque()
        self.log = deque(maxlen=RING_MAX)
        self.seq = 0                  # bumps on every change; the heartbeat
                                      # carries it so the page knows to refetch
        self._per_window = int(per_window)
        self._window_s = float(window_s)
        self._global_s = global_s
        self._last_greet = 0.0
        self._n = 0
        self._budget_path = Path(budget_path)
        self._budget = {}             # ip -> [timestamps]
        self._load_budget()

    # -- the budget --------------------------------------------------------
    # Persisted on purpose. A budget that lives only in memory is a budget a
    # restart refills, and this process restarts whenever the fish does.
    def _load_budget(self):
        try:
            raw = json.loads(self._budget_path.read_text())
        except (OSError, ValueError):
            return
        now = time.time()
        if isinstance(raw, dict):
            self._budget = {k: [float(t) for t in v if now - float(t) < self._window_s]
                            for k, v in raw.items() if isinstance(v, list)}
            self._budget = {k: v for k, v in self._budget.items() if v}

    def _save_budget(self):
        """Caller holds the lock. Prunes by age on the way out, so the file is
        bounded by live visitors rather than by everyone who ever visited."""
        now = time.time()
        self._budget = {k: [t for t in v if now - t < self._window_s]
                        for k, v in self._budget.items()}
        self._budget = {k: v for k, v in self._budget.items() if v}
        try:
            self._budget_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._budget_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._budget))
            os.replace(tmp, self._budget_path)
        except OSError as exc:
            print(f"show budget: {exc}")

    # -- in ----------------------------------------------------------------
    def submit(self, text, ip="?"):
        """(ok, detail, http_code). Refusals are phrased for a human to read,
        and carry their own status so the endpoint never has to guess one back
        out of the wording."""
        clean, why = sanitise(text, self.words)
        if why:
            return False, why, 400
        now = time.time()
        with self.lock:
            if len(self.inbox) >= INBOX_MAX:
                return False, "it already has a queue; try again in a minute", 429
            spent = [t for t in self._budget.get(ip, []) if now - t < self._window_s]
            if len(spent) >= self._per_window:
                mins = int((self._window_s - (now - min(spent))) / 60)
                return False, (f"{self._per_window} per 6 hours, and you have "
                               f"used them — {mins} min to go"), 429
            spent.append(now)
            self._budget[ip] = spent
            self._save_budget()
            self._n += 1
            self.inbox.append({"id": self._n, "text": clean, "at": round(now, 2)})
            self.seq += 1
            return True, len(self.inbox), 200

    def next_message(self):
        """The next message the fish should be shown, or None — None also when
        the last greeting was too recent, so browsing stays the default."""
        with self.lock:
            if not self.inbox or time.time() - self._last_greet < self._global_s:
                return None
            self._last_greet = time.time()
            return self.inbox.popleft()

    # -- out ---------------------------------------------------------------
    def finish(self, msg, reaction):
        entry = {"id": msg["id"], "text": msg["text"], "at": msg["at"],
                 "done": round(time.time(), 2), "reaction": reaction,
                 "said": list(reaction.get("said") or []),
                 "line": line(reaction)}
        with self.lock:
            self.log.append(entry)
            self.seq += 1
        return entry

    def ring(self):
        with self.lock:
            return [dict(e) for e in reversed(self.log)]    # newest first

    def state(self):
        with self.lock:
            return {"seq": self.seq, "queued": len(self.inbox),
                    "max_words": MAX_WORDS, "per_6h": self._per_window,
                    "list": len(self.words)}
