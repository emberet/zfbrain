"""Letting visitors show the fish something, and reading back what it did.

The site says, in `#honest`, that the fish has no language. That is still true,
so this is not a chat box. A visitor's message is *drawn* onto a page-sized
canvas and drifted across the retina as a moving stimulus; the reply is the
measured reaction of the same 187,053 neurons that read every other page —
DSGC, nMLF, vSPN, Mauthner rates, and whether a bout, a turn or a startle came
out of the decode. The retina is 18x12, so a message arrives as 216 numbers.

Three layers, most honest first:

  1. the reaction itself, in Hz, which is the fish's actual answer
  2. `line()` — a deterministic transcription of those numbers, built by
     template. No model, no API key, never invents anything. This is why the
     feature works with ANTHROPIC_API_KEY unset.
  3. a narrator sentence from voice.py's existing number-checked pipeline,
     filled in afterwards on a background thread and clearly labelled on the
     page as the narrator's. Absent, silently, when there is no key.

Nothing here touches the browser: no navigation, no click, no mouse. The
message becomes light on a retina and nothing else, which is also why an
arbitrary visitor string cannot reach a real web page.
"""

import os
import re
import threading
import time
from collections import deque

VIEW_W, VIEW_H = 1280, 800     # the fish's viewport, same as roam.py's

MAX_CHARS = 120
INBOX_MAX = 8                  # waiting messages; over this, politely full
RING_MAX = 8                   # finished exchanges the site shows
PER_IP_S = 60.0                # one message per visitor per minute
GLOBAL_S = 20.0                # ...and the fish gets at least this long between
                               # greetings, so it still spends its life browsing

# What a message may contain. Deliberately narrow: this is drawn onto a public
# live video feed, so the set is letters, digits, space and the punctuation you
# need for a sentence — nothing that can look like markup, a path or a command.
ALLOWED = re.compile(r"^[A-Za-z0-9 .,!?'\-:;()]+$")
URLISH = re.compile(r"://|\bwww\.|\.(com|net|org|io|xyz|co|ru|cn)\b", re.I)

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


def sanitise(text):
    """(clean, reason). `reason` is None when the message is usable, otherwise
    a sentence the endpoint hands straight back to the visitor."""
    if not isinstance(text, str):
        return None, "a message has to be text"
    clean = " ".join(text.split())          # collapse newlines and runs of space
    if not clean:
        return None, "nothing to show it"
    if len(clean) > MAX_CHARS:
        return None, f"too long — {MAX_CHARS} characters at most"
    if URLISH.search(clean):
        return None, "no links: it cannot follow one, and this is a public feed"
    if not ALLOWED.match(clean):
        return None, "letters, digits and simple punctuation only"
    return clean, None


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
    do, which is the whole reason it exists alongside the narrator."""
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
    return (f"{head} {reaction.get('mean_hz', 0.0):.1f} Hz across the graph over "
            f"{reaction.get('thoughts', 0)} thoughts.")


class Talk:
    """The inbox, the transcript ring, and the rate limits between them.

    One lock covers all of it. The roamer's sim thread is the only caller of
    `next_message`/`finish`; the HTTP threads call `submit`/`ring`; the narrator
    thread mutates finished exchanges in place. Everything that leaves goes out
    as a copy, so no caller can hold a reference into the ring."""

    def __init__(self, narrator=None, per_ip_s=PER_IP_S, global_s=GLOBAL_S):
        self.lock = threading.Lock()
        self.inbox = deque()
        self.log = deque(maxlen=RING_MAX)
        self.seq = 0                  # bumps on every change; the heartbeat
                                      # carries it so the page knows to refetch
        self._ip_at = {}
        self._last_greet = 0.0
        self._per_ip_s = per_ip_s
        self._global_s = global_s
        self._n = 0
        self._narrator = narrator     # voice.Voice, or None
        self._pending = deque()
        self._wake = threading.Event()
        if narrator is not None:
            threading.Thread(target=self._narrate_loop, daemon=True).start()

    # -- in ----------------------------------------------------------------
    def submit(self, text, ip="?"):
        """(ok, detail, http_code). Refusals are phrased for a human to read,
        and carry their own status so the endpoint never has to guess one back
        out of the wording."""
        clean, why = sanitise(text)
        if why:
            return False, why, 400
        now = time.time()
        with self.lock:
            if len(self.inbox) >= INBOX_MAX:
                return False, "it already has a queue; try again in a minute", 429
            last = self._ip_at.get(ip, 0.0)
            if now - last < self._per_ip_s:
                return False, f"one at a time — {int(self._per_ip_s - (now - last))}s to go", 429
            if len(self._ip_at) > 4096:           # a bounded memory, not a cache
                self._ip_at = {k: v for k, v in self._ip_at.items()
                               if now - v < self._per_ip_s}
            self._ip_at[ip] = now
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
                 "line": line(reaction), "narrator": None}
        with self.lock:
            self.log.append(entry)
            self.seq += 1
        if self._narrator is not None:
            self._pending.append(entry)
            self._wake.set()
        return entry

    def ring(self):
        with self.lock:
            return [dict(e) for e in reversed(self.log)]    # newest first

    def state(self):
        with self.lock:
            return {"seq": self.seq, "queued": len(self.inbox),
                    "max_chars": MAX_CHARS}

    # -- the narrator ------------------------------------------------------
    def _narrate_loop(self):
        """Layer 3, off the sim thread on purpose: an API call takes a second or
        two and the brain must not wait for it. A failure here leaves `narrator`
        None and the deterministic line standing."""
        while True:
            self._wake.wait(timeout=30)
            self._wake.clear()
            while self._pending:
                entry = self._pending.popleft()
                try:
                    text = self._narrator.reply({"message": entry["text"],
                                                 "reaction": entry["reaction"]})
                except Exception as exc:  # noqa: BLE001 — never kill the thread
                    print(f"talk: narrator failed ({exc})")
                    text = None
                if text:
                    with self.lock:
                        entry["narrator"] = text
                        self.seq += 1
