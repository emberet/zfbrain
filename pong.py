"""A table the fish can actually play on, and an honest account of how.

WHAT THE FISH CONTRIBUTES, AND WHAT IT DOES NOT
-----------------------------------------------
Read this before the code, because the interesting part of this file is the
part that had to be given up.

The obvious design - draw a ball, let the fish watch it, move its paddle toward
it - does not work, and `tools/probe_flow.py` is the measurement that killed it.
A ball on a plain field drives the direction-selective ganglion cells to a
decoded asymmetry of |dy| <= 0.016 at every radius tried (40, 90, 150 px),
against the 0.08 needed to move anything. Worse, a hard disc on a flat
background is a whole-field luminance step - a *flash* - and it tripped Mauthner
into an escape on 7 of 12 frames. The fish does not chase the ball. It flinches
away from it.

What does drive this retina, 10x over the gate, is a drifting grating:

    still grating          |dy| 0.004     0/8  escapes
    grating +218 px/frame  |dy| 0.777     0/12 escapes
    grating -218 px/frame  |dy| 0.780     0/12 escapes
    ball alone (r=150)     |dy| 0.016     4/12 escapes
    ball over a grating    |dy| 0.613     0/15 escapes

218 px is not tuned. `Retina._motion` is a displaced-frame difference at
shift=3, so it is a matched filter that nulls when the image translates by
exactly 3 sample rows, and the 12 sample rows of an 800 px viewport sit 72.6 px
apart. The discriminator is not the peak in `down` - that channel clips at
motion_max and saturates for any drift - it is the *null* in `up`.

So this is an optomotor rig, which is what a real lab would build. You put a
larva in a dish, drift a grating under it, and it swims to hold station. The
experimenter picks the grating's direction. Here the experimenter is one line of
Python - `grating_dir()` - which points the grating the way the ball lies
relative to the paddle. The fish supplies the reflex and nothing else: grating
direction in, paddle velocity out. It is not tracking the ball. It is holding
station against a moving floor that someone else aimed.

That is a real sensorimotor loop and a real behaviour - the fish's own paddle is
drawn into the scene it sees, so its motion feeds back through its own retina -
but the error signal is computed outside the brain, and the site says so at the
same size as everything else. `rig=False` renders the raw ball instead, and is
the feature's own falsification: |dy| falls from 0.78 to 0.004 and the paddle
stops. Anyone can flip it and watch the claim fail.

Nothing here is a reward. The fish is never told the score, never told whether
it returned the ball, and nothing about winning reaches a synapse. It would
drift after a moving floor in an empty room in exactly the same way.

Nothing visitor-supplied is ever drawn or interpreted. The only thing a visitor
controls is one clamped float - the y of their own paddle - so there is no
parser, no text, and no injection surface anywhere on this path.
"""

import secrets
import threading
import time
from collections import deque

import numpy as np

VIEW_W, VIEW_H = 1280, 800          # the fish's viewport, same as roam.py's

# -- the geometry the retina dictates ------------------------------------
# Retina._downsample point-samples 12 rows out of 800 via linspace, so these
# are the exact pixel rows the fish can see. Everything below is derived from
# them rather than chosen; tools/probe_flow.py is the measurement that checks
# the derivation against the real brain.
ROWS = np.linspace(0, VIEW_H - 1, 12).astype(int)
DRIFT_PX = int(ROWS[4] - ROWS[1])   # 3 sample rows = 218 px: the matched filter
# 6 sample rows apart must land antiphase or the `down` channel nulls as well,
# so the period is 2/3 of that span. A sinusoid, not a square wave - the retina
# point-samples, and square edges alias into whichever channel gets lucky.
GRATING_P = 2.0 * float(ROWS[7] - ROWS[1]) / 3.0     # ~290.7 px

# A page is mostly light and the brain is calibrated against that regime; a dark
# table would read as a dim room and the fish would simply go quiet.
BG = 232
GRATING_LO, GRATING_HI = 40.0, 215.0
INK = 18                            # ball and paddles: dark on light, like text

# -- the table ------------------------------------------------------------
PADDLE_H = 150
PADDLE_W = 18
PADDLE_INSET = 60                   # x of each paddle's outer face
BALL_R = 26
BALL_VX = 200.0                     # px per thought: ~6.4 thoughts to cross
BALL_VY_MAX = 150.0
PADDLE_GAIN = 150.0                 # px of paddle travel per unit of decoded dy
VISITOR_MAX_DY = 320.0              # px per thought, so a human can still miss
DEADBAND = 40.0                     # closer than this and the grating holds still

# -- seat control ---------------------------------------------------------
SEAT_IDLE_S = 20.0                  # silence that gives the table up
PER_IP_S = 30.0                     # one join per visitor per half minute
RING_MAX = 8                        # finished points the site shows

_Y = np.arange(VIEW_H, dtype=np.float32)[:, None]


def _grating(phase):
    """One drifting sinusoid, full field, as uint8. This is the stimulus that
    actually moves the fish; the ball is painted on top of it."""
    band = GRATING_LO + (GRATING_HI - GRATING_LO) * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * (_Y - phase) / GRATING_P))
    return np.repeat(band.astype(np.uint8), VIEW_W, axis=1)


def render(state):
    """The fish's whole view, 1280x800, as a PIL image.

    Drawn, never HTML - there is no parser anywhere on this path. The visitor's
    own paddle is here too, but only as a rectangle at a clamped y; nothing a
    visitor typed has ever reached this function, because there is no way for a
    visitor to type anything at all.
    """
    from PIL import Image, ImageDraw

    if state.get("rig", True):
        arr = _grating(state.get("phase", 0.0))
    else:
        # the falsification: the same table with no rig behind it. The retina
        # sees a flat field and a small hard disc, which measured |dy| 0.004.
        arr = np.full((VIEW_H, VIEW_W), BG, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    d = ImageDraw.Draw(img)

    for x, y in ((PADDLE_INSET, state["fish_y"]),
                 (VIEW_W - PADDLE_INSET - PADDLE_W, state["visitor_y"])):
        d.rectangle([x, y - PADDLE_H / 2, x + PADDLE_W, y + PADDLE_H / 2],
                    fill=(INK, INK, INK))
    bx, by = state["ball_x"], state["ball_y"]
    d.ellipse([bx - BALL_R, by - BALL_R, bx + BALL_R, by + BALL_R],
              fill=(INK, INK, INK))
    return img


def line(point):
    """One finished point, as a sentence, by template.

    Every number in it is one the rally actually produced. Like talk.line()
    before it, this cannot say anything the fish did not do - there is no model
    behind it and no API key, so it cannot invent a good game.
    """
    n, who = point.get("rally", 0), point.get("winner", "?")
    dy = point.get("mean_dy", 0.0)
    if not point.get("rig", True):
        return (f"No rig: the grating was off, the fish saw a ball on a flat field, "
                f"and its decoded drift averaged {dy:.3f}. {n} return"
                f"{'' if n == 1 else 's'} — the paddle barely moved, which is the point.")
    if n == 0:
        head = "It never got to the ball."
    elif n == 1:
        head = "One return."
    else:
        head = f"{n} returns before the point ended."
    tail = "the visitor took it" if who == "visitor" else "the fish took it"
    return f"{head} Mean decoded drift {dy:.3f}; {tail}."


class Table:
    """Ball, both paddles, the score, and the seat, under one lock.

    The roamer's sim thread is the only caller of `advance`, `grating_dir` and
    `move_fish`; HTTP threads call `join`, `move` and `state`. Everything that
    leaves goes out as a copy, so no caller can hold a reference into the ring.
    """

    def __init__(self, rig=True, seat_idle_s=SEAT_IDLE_S, per_ip_s=PER_IP_S):
        self.lock = threading.Lock()
        self.seq = 0                   # bumps on every change; the heartbeat
                                       # carries it so the page knows to refetch
        self.log = deque(maxlen=RING_MAX)
        self.rig = bool(rig)
        self.phase = 0.0
        self._dir = 0
        self._seat = None              # token of whoever holds the table
        self._seat_at = 0.0
        self._seat_ip = None
        self._ip_at = {}
        self._per_ip_s = per_ip_s
        self._idle_s = seat_idle_s
        self._dys = []
        self._n = 0
        self.score = {"fish": 0, "visitor": 0}
        self.rally = 0
        self._serve(+1)

    # -- geometry helpers --------------------------------------------------
    @staticmethod
    def _clampy(y):
        return float(min(max(y, PADDLE_H / 2), VIEW_H - PADDLE_H / 2))

    def _serve(self, toward):
        """Put the ball back in the middle heading `toward` (-1 fish, +1 visitor)."""
        rng = np.random.default_rng(self._n * 7919 + 13)
        self.ball_x = VIEW_W / 2.0
        self.ball_y = float(rng.uniform(BALL_R * 2, VIEW_H - BALL_R * 2))
        self.ball_vx = BALL_VX * toward
        self.ball_vy = float(rng.uniform(-BALL_VY_MAX, BALL_VY_MAX))
        self.fish_y = VIEW_H / 2.0
        self.visitor_y = VIEW_H / 2.0
        self.rally = 0
        self._dys = []

    # -- the rig -----------------------------------------------------------
    def grating_dir(self):
        """Which way to drift the floor: +1 down, -1 up, 0 hold.

        This is the line that is not the fish. It computes where the ball is
        relative to the fish's paddle and aims the grating there; the brain
        never sees this number, only the moving image that results. Called on
        the sim thread once per thought, before the frame is rendered.
        """
        with self.lock:
            if not self.rig:
                return 0
            err = self.ball_y - self.fish_y
            return 0 if abs(err) < DEADBAND else (1 if err > 0 else -1)

    def drift(self, direction):
        """Advance the grating by exactly one matched-filter step."""
        with self.lock:
            self._dir = int(direction)
            self.phase += DRIFT_PX * self._dir
            self.seq += 1

    def move_fish(self, dy):
        """Move the fish's paddle by its own decoded asymmetry, and nothing else.

        `dy` is decode()['dy'] - (down - up) / (down + up) over the DSGC pool.
        Positive means downward flow, which is the optomotor response to a floor
        drifting down, and the paddle follows it. That is the entire motor path.
        """
        with self.lock:
            self.fish_y = self._clampy(self.fish_y + float(dy) * PADDLE_GAIN)
            self._dys.append(abs(float(dy)))
            self.seq += 1

    # -- physics -----------------------------------------------------------
    def advance(self):
        """One thought of ball. Returns an event string, or None.

        'return_fish' / 'return_visitor' when a paddle got there, 'point_fish' /
        'point_visitor' when one did not. The fish is never told which.
        """
        with self.lock:
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy
            if self.ball_y < BALL_R:
                self.ball_y, self.ball_vy = BALL_R, abs(self.ball_vy)
            elif self.ball_y > VIEW_H - BALL_R:
                self.ball_y, self.ball_vy = VIEW_H - BALL_R, -abs(self.ball_vy)
            self.seq += 1

            fish_face = PADDLE_INSET + PADDLE_W
            vis_face = VIEW_W - PADDLE_INSET - PADDLE_W
            if self.ball_vx < 0 and self.ball_x <= fish_face + BALL_R:
                if abs(self.ball_y - self.fish_y) <= PADDLE_H / 2 + BALL_R:
                    return self._bounce(fish_face + BALL_R, +1, self.fish_y, "return_fish")
                return self._point("visitor")
            if self.ball_vx > 0 and self.ball_x >= vis_face - BALL_R:
                if abs(self.ball_y - self.visitor_y) <= PADDLE_H / 2 + BALL_R:
                    return self._bounce(vis_face - BALL_R, -1, self.visitor_y,
                                        "return_visitor")
                return self._point("fish")
            return None

    def _bounce(self, x, toward, paddle_y, event):
        """Caller holds the lock. Angle off the paddle face, as pong has always
        done it: the further from the middle of the bat, the steeper the return."""
        self.ball_x = x
        self.ball_vx = BALL_VX * toward
        off = (self.ball_y - paddle_y) / (PADDLE_H / 2)
        self.ball_vy = float(np.clip(off, -1.0, 1.0)) * BALL_VY_MAX
        self.rally += 1
        return event

    def _point(self, winner):
        """Caller holds the lock."""
        self.score[winner] += 1
        self._n += 1
        mean_dy = float(np.mean(self._dys)) if self._dys else 0.0
        entry = {"id": self._n, "rally": self.rally, "winner": winner,
                 "mean_dy": round(mean_dy, 4), "rig": self.rig,
                 "at": round(time.time(), 2),
                 "score": dict(self.score)}
        entry["line"] = line(entry)
        self.log.append(entry)
        self._serve(-1 if winner == "fish" else +1)
        self.seq += 1
        return f"point_{winner}"

    # -- the seat ----------------------------------------------------------
    def _expired(self, now):
        """Caller holds the lock."""
        return self._seat is not None and now - self._seat_at > self._idle_s

    def join(self, ip="?"):
        """(ok, token-or-reason, http code). One visitor at a time; everyone
        else watches the same rally over the stream that already exists."""
        now = time.time()
        with self.lock:
            if self._expired(now):
                self._seat = None
            if self._seat is not None:
                left = int(self._idle_s - (now - self._seat_at))
                return False, f"someone is at the table — {max(left, 1)}s left", 409
            last = self._ip_at.get(ip, 0.0)
            if now - last < self._per_ip_s:
                return False, f"just a moment — {int(self._per_ip_s - (now - last))}s", 429
            if len(self._ip_at) > 4096:       # a bounded memory, not a cache
                self._ip_at = {k: v for k, v in self._ip_at.items()
                               if now - v < self._per_ip_s}
            self._ip_at[ip] = now
            self._seat = secrets.token_urlsafe(16)
            self._seat_at = now
            self._seat_ip = ip
            self.seq += 1
            return True, self._seat, 200

    def move(self, token, y):
        """The only thing a visitor can do. `y` is clamped to the table before
        it is stored and is never used for anything but a rectangle."""
        now = time.time()
        try:
            y = float(y)
        except (TypeError, ValueError):
            return False, "y has to be a number", 400
        if not np.isfinite(y):
            return False, "y has to be a number", 400
        with self.lock:
            if self._seat is None or self._expired(now):
                self._seat = None
                return False, "nobody is holding the table — join first", 409
            if not isinstance(token, str) or not secrets.compare_digest(token, self._seat):
                return False, "that is not your seat", 403
            self._seat_at = now
            want = self._clampy(y)
            step = float(np.clip(want - self.visitor_y, -VISITOR_MAX_DY, VISITOR_MAX_DY))
            self.visitor_y = self._clampy(self.visitor_y + step)
            self.seq += 1
            return True, self.visitor_y, 200

    def release(self, token=None):
        """Free the seat. `token=None` frees it unconditionally and is for the
        rally path only: roam.py's /pong/leave refuses a request whose token is
        not a string, because otherwise an empty POST boots whoever is playing.
        """
        with self.lock:
            if token is None or (self._seat and secrets.compare_digest(token, self._seat)):
                self._seat, self._seat_ip = None, None
                self.seq += 1
                return True
            return False

    def occupied(self):
        with self.lock:
            if self._expired(time.time()):
                self._seat = None
            return self._seat is not None

    def set_rig(self, on):
        """Flip the rig off to watch the claim fail. Only between points, so a
        rally is never half one experiment and half the other."""
        with self.lock:
            self.rig = bool(on)
            self.seq += 1
            return self.rig

    # -- readout -----------------------------------------------------------
    def snapshot(self):
        """What render() needs, taken under the lock in one go."""
        with self.lock:
            return {"ball_x": self.ball_x, "ball_y": self.ball_y,
                    "fish_y": self.fish_y, "visitor_y": self.visitor_y,
                    "phase": self.phase, "rig": self.rig}

    def state(self):
        """Small enough to ride the heartbeat several times a second."""
        with self.lock:
            held = self._seat is not None and not self._expired(time.time())
            return {"seq": self.seq, "rig": self.rig, "held": held,
                    "ball": [round(self.ball_x, 1), round(self.ball_y, 1)],
                    "fish_y": round(self.fish_y, 1),
                    "visitor_y": round(self.visitor_y, 1),
                    "drift": self._dir, "rally": self.rally,
                    "score": dict(self.score),
                    # the geometry too, so the page can draw the table from the
                    # feed instead of keeping its own copy of these constants
                    # and drifting out of step with the image the fish sees
                    "w": VIEW_W, "h": VIEW_H, "paddle_h": PADDLE_H,
                    "paddle_w": PADDLE_W, "inset": PADDLE_INSET,
                    "ball_r": BALL_R}

    def ring(self):
        with self.lock:
            return [dict(e) for e in reversed(self.log)]      # newest first


def smoke():
    """Everything here is cheap and deterministic - no brain, no network.

        python pong.py
    """
    # -- the geometry the whole feature rests on --------------------------
    assert DRIFT_PX == 218, f"the matched filter moved: {DRIFT_PX}"
    assert abs(GRATING_P - 290.666) < 0.01, f"grating period drifted: {GRATING_P}"

    # -- render is the fish's viewport, exactly ---------------------------
    t = Table()
    img = render(t.snapshot())
    assert img.size == (VIEW_W, VIEW_H), f"render is {img.size}, not the viewport"
    assert render({**t.snapshot(), "rig": False}).size == (VIEW_W, VIEW_H)

    # the rig has to actually put a moving grating in front of the retina: two
    # frames one DRIFT_PX apart must differ, and a still one must not
    a = np.asarray(render({**t.snapshot(), "phase": 0.0}).convert("L"), dtype=float)
    b = np.asarray(render({**t.snapshot(), "phase": DRIFT_PX}).convert("L"), dtype=float)
    c = np.asarray(render({**t.snapshot(), "phase": 0.0}).convert("L"), dtype=float)
    assert np.abs(a - b).mean() > 20.0, "the grating did not move"
    assert np.array_equal(a, c), "render is not deterministic in phase"

    # -- the rig aims at the ball, and holds still inside the deadband ----
    t.fish_y, t.ball_y = 400.0, 700.0
    assert t.grating_dir() == +1, "ball below the paddle should drift the floor down"
    t.ball_y = 100.0
    assert t.grating_dir() == -1, "ball above the paddle should drift the floor up"
    t.ball_y = 400.0 + DEADBAND / 2
    assert t.grating_dir() == 0, "inside the deadband the floor holds still"
    t.rig = False
    assert t.grating_dir() == 0, "with the rig off there is no grating to aim"
    t.rig = True

    # -- the paddle clamps to the table, both ends ------------------------
    t.fish_y = VIEW_H / 2
    for _ in range(40):
        t.move_fish(+1.0)
    assert t.fish_y == VIEW_H - PADDLE_H / 2, f"fish paddle left the table: {t.fish_y}"
    for _ in range(80):
        t.move_fish(-1.0)
    assert t.fish_y == PADDLE_H / 2, f"fish paddle left the table: {t.fish_y}"

    # -- seat control ------------------------------------------------------
    t2 = Table(seat_idle_s=0.2, per_ip_s=0.0)
    ok, tok, code = t2.join("1.2.3.4")
    assert ok and code == 200 and isinstance(tok, str)
    ok2, why, code2 = t2.join("5.6.7.8")
    assert not ok2 and code2 == 409, f"a second visitor got in: {why}"
    ok3, _, code3 = t2.move("not-the-token", 300)
    assert not ok3 and code3 == 403, "a wrong token moved the paddle"
    ok4, y, _ = t2.move(tok, 300.0)
    assert ok4 and abs(y - 300.0) < 1e-6, f"the seat holder could not move: {y}"
    # a move past the per-thought limit is clamped, not refused
    ok5, y5, _ = t2.move(tok, VIEW_H)
    assert ok5 and y5 <= 300.0 + VISITOR_MAX_DY + 1e-6, f"visitor teleported: {y5}"
    # ...and a non-number is refused before it reaches any of that
    assert t2.move(tok, "700; DROP")[2] == 400
    assert t2.move(tok, float("nan"))[2] == 400
    time.sleep(0.25)                       # the seat times out on silence
    assert not t2.occupied(), "the seat outlived its idle timeout"
    ok6, tok6, _ = t2.join("5.6.7.8")
    assert ok6, "the table never freed up"
    assert t2.release(tok6) and not t2.occupied()

    # -- per-IP limit ------------------------------------------------------
    t3 = Table(seat_idle_s=0.05, per_ip_s=60.0)
    ok, tok, _ = t3.join("9.9.9.9")
    assert ok
    t3.release(tok)
    assert t3.join("9.9.9.9")[2] == 429, "the same IP re-took the table at once"

    # -- physics: a point is won, a rally is counted ----------------------
    t4 = Table()
    t4.ball_x, t4.ball_y, t4.ball_vx, t4.ball_vy = 300.0, 400.0, -BALL_VX, 0.0
    t4.fish_y = 400.0                       # squarely behind it
    assert t4.advance() == "return_fish", "the fish's paddle missed a ball on its face"
    assert t4.rally == 1 and t4.ball_vx > 0
    t4.ball_x, t4.ball_vx = 300.0, -BALL_VX
    t4.fish_y = 50.0 + PADDLE_H / 2         # nowhere near it
    assert t4.advance() == "point_visitor"
    assert t4.score["visitor"] == 1 and t4.rally == 0
    assert t4.ring()[0]["rally"] == 1, "the finished point did not reach the ring"

    # -- the ball stays on the table --------------------------------------
    t5 = Table()
    t5.ball_vy = BALL_VY_MAX
    for _ in range(200):
        t5.advance()
        assert BALL_R - 1 <= t5.ball_y <= VIEW_H - BALL_R + 1, f"ball left: {t5.ball_y}"

    # -- line() says only what happened -----------------------------------
    s = line({"rally": 3, "winner": "fish", "mean_dy": 0.62, "rig": True})
    assert "3 returns" in s and "0.620" in s and "fish took it" in s, s
    assert "no rig" in line({"rally": 0, "winner": "visitor", "mean_dy": 0.004,
                             "rig": False}).lower()

    # -- state() is small enough to ride the heartbeat --------------------
    import json
    assert len(json.dumps(t.state())) < 400, "pong state is too fat for the SSE feed"
    print("pong smoke OK")


if __name__ == "__main__":
    smoke()
