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

# How much of the time the rig is allowed to aim. The floor drifting is the only
# thing that moves the fish's paddle - a still floor decodes |dy| ~0.01, well
# under the 0.08 it takes to move anything - so this fraction *is* the fish's
# difficulty, and there is no separate skill knob hiding behind it.
#
# tools/rally.py --fair, 10 rallies x 40 thoughts against a tracking visitor:
#
#     aim    |dy|   paddle-ball   fish win%
#    1.00   0.566      89.5 px      59.5%
#    0.75   0.435     111.4 px      65.1%
#    0.50   0.299     122.7 px      48.1%
#    0.35   0.227     144.0 px      48.2%
#    0.25   0.173     154.1 px      46.0%
#    0.00   0.005     178.1 px      47.8%
#
# Not chosen on win%, which is noise: at aim 0.00 the rig is off entirely and
# the paddle barely moves, yet the fish still takes 47.8% - that column is
# measuring the opponent missing, not the fish playing. Chosen on the two
# columns that are monotonic and mean something:
#
#   * paddle-ball must exceed the paddle's own reach, PADDLE_H/2 + BALL_R =
#     101 px, or the fish is within range of the ball on average and looks
#     like it is auto-hitting. That rules out 1.00, and 0.75 only just clears.
#   * |dy| must stay well clear of the 0.08 that moves anything, or the paddle
#     stops being visibly the fish's doing and the whole section loses its
#     point.
#
# 0.50 clears both with margin: 122.7 px is a fifth outside reach, and 0.299 is
# nearly four times the threshold.
AIM_DUTY = 0.5

# -- the rally score, after Chrome's dinosaur ----------------------------
# The dino scores distance survived, not goals, and keeps a HI next to it. That
# suits this table better than fish-vs-visitor does: the interesting quantity is
# how long the two of you kept the ball alive, and the fish is not trying to win
# anyway. Ticks while the ball is in play, jumps on a return, resets when the
# point ends. HI lives in the process - a restart forgets it, and the page says
# so rather than pretending otherwise.
SCORE_PER_THOUGHT = 16              # one per sub-step at RALLY_SUBSTEPS=16
SCORE_PER_RETURN = 100

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
    run = point.get("run_score")
    score = "" if run is None else (
        f" {run:05d}{' — new best' if point.get('best') else ''}.")
    return f"{head} Mean decoded drift {dy:.3f}; {tail}.{score}"


class Table:
    """Ball, both paddles, the score, and the seat, under one lock.

    The roamer's sim thread is the only caller of `advance`, `grating_dir` and
    `move_fish`; HTTP threads call `join`, `move` and `state`. Everything that
    leaves goes out as a copy, so no caller can hold a reference into the ring.
    """

    def __init__(self, rig=True, seat_idle_s=SEAT_IDLE_S, per_ip_s=PER_IP_S,
                 aim=AIM_DUTY):
        self.lock = threading.Lock()
        self.seq = 0                   # bumps on every change; the heartbeat
                                       # carries it so the page knows to refetch
        self.log = deque(maxlen=RING_MAX)
        self.rig = bool(rig)
        self.aim = float(min(max(aim, 0.0), 1.0))
        self.phase = 0.0
        self._dir = 0
        self._credit = 0.0             # the duty cycle's accumulator, not an RNG
        self._seat = None              # token of whoever holds the table
        self._seat_at = 0.0
        self._seat_ip = None
        self._ip_at = {}
        self._per_ip_s = per_ip_s
        self._idle_s = seat_idle_s
        self._dys = []
        self._n = 0
        self.score = {"fish": 0, "visitor": 0}
        self.run_score = 0.0           # this point, dino-style
        self.hi_score = 0              # best since the process started
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
        self.run_score = 0.0           # the dino reset: a miss costs the run
        self._dys = []

    # -- the rig -----------------------------------------------------------
    def grating_dir(self):
        """Which way to drift the floor: +1 down, -1 up, 0 hold.

        This is the line that is not the fish. It computes where the ball is
        relative to the fish's paddle and aims the grating there; the brain
        never sees this number, only the moving image that results. Called on
        the sim thread once per thought, before the frame is rendered.

        `aim` withholds the rig on some thoughts. A duty cycle rather than a
        weaker push, because the push has no strength to weaken: the grating
        steps by exactly one matched-filter lag or it does not step at all, and
        a smaller step would land off the null in `up` and drive both channels.
        So the only thing that can be dialled is *how often*, and this dials it.

        Deterministic - a Bresenham accumulator, not an RNG - because the fish
        is a closed loop and a random floor would be a different experiment on
        every rally. At aim=0.5 the rig aims on alternate thoughts, and the
        withheld ones are the fish coasting on its own last decode.
        """
        with self.lock:
            if not self.rig:
                return 0
            err = self.ball_y - self.fish_y
            if abs(err) < DEADBAND:
                return 0
            self._credit += self.aim
            if self._credit < 1.0:
                return 0
            self._credit -= 1.0
            return 1 if err > 0 else -1

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
    def advance(self, frac=1.0):
        """`frac` of a thought of ball. Returns an event string, or None.

        'return_fish' / 'return_visitor' when a paddle got there, 'point_fish' /
        'point_visitor' when one did not. The fish is never told which.

        The rally calls this in sub-steps while the brain is thinking, so the
        ball crosses the table smoothly instead of teleporting 200 px once a
        second. That is purely how the ball is drawn between retina samples -
        the grating still steps exactly one matched-filter lag per thought, and
        the retina still samples exactly once per thought, so nothing here
        changes what the fish is shown or how it decodes it. Summing frac to
        1.0 over a thought reproduces the old single call.
        """
        with self.lock:
            self.run_score += SCORE_PER_THOUGHT * frac
            self.ball_x += self.ball_vx * frac
            self.ball_y += self.ball_vy * frac
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
        self.run_score += SCORE_PER_RETURN
        return event

    def _point(self, winner):
        """Caller holds the lock."""
        self.score[winner] += 1
        self._n += 1
        mean_dy = float(np.mean(self._dys)) if self._dys else 0.0
        run = int(self.run_score)
        best = run > self.hi_score
        self.hi_score = max(self.hi_score, run)
        entry = {"id": self._n, "rally": self.rally, "winner": winner,
                 "mean_dy": round(mean_dy, 4), "rig": self.rig,
                 "at": round(time.time(), 2), "run_score": run, "best": best,
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
            return {"seq": self.seq, "rig": self.rig, "aim": self.aim, "held": held,
                    "ball": [round(self.ball_x, 1), round(self.ball_y, 1)],
                    "fish_y": round(self.fish_y, 1),
                    "visitor_y": round(self.visitor_y, 1),
                    "drift": self._dir, "rally": self.rally,
                    "score": dict(self.score),
                    "run_score": int(self.run_score), "hi_score": self.hi_score,
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
    full = Table(aim=1.0)
    full.fish_y, full.ball_y = 400.0, 700.0
    assert full.grating_dir() == +1, "ball below the paddle should drift the floor down"
    full.ball_y = 100.0
    assert full.grating_dir() == -1, "ball above the paddle should drift the floor up"
    full.ball_y = 400.0 + DEADBAND / 2
    assert full.grating_dir() == 0, "inside the deadband the floor holds still"
    full.rig = False
    assert full.grating_dir() == 0, "with the rig off there is no grating to aim"

    # -- the duty cycle withholds the rig, and does it deterministically --
    # the deadband is checked first, so a ball this far out is always aimable
    # and every 0 below is the duty cycle and not the deadband
    for aim, want in ((1.0, 12), (0.5, 6), (0.25, 3), (0.0, 0)):
        d = Table(aim=aim)
        d.fish_y, d.ball_y = 400.0, 700.0
        fired = [d.grating_dir() for _ in range(12)]
        assert sum(abs(x) for x in fired) == want, (aim, fired)
        assert set(fired) <= {0, +1}, f"aim must not flip the sign: {fired}"
        again = Table(aim=aim)
        again.fish_y, again.ball_y = 400.0, 700.0
        assert [again.grating_dir() for _ in range(12)] == fired, "duty cycle is not deterministic"
    assert Table(aim=5.0).aim == 1.0 and Table(aim=-1.0).aim == 0.0, "aim must clamp to [0,1]"

    # -- sub-stepping must not change where the ball goes -----------------
    # roam.py cuts a thought into RALLY_SUBSTEPS pieces so the ball is drawn
    # moving instead of teleporting. If the pieces do not sum to the whole, the
    # ball quietly changes speed and the 6.4-thoughts-to-cross figure the whole
    # section quotes stops being true.
    # Free flight only. Where a paddle or a wall is involved the two *should*
    # differ - a sub-step catches the face before the ball has overshot it, so
    # the bounce angle is off a different part of the bat. That is finer
    # collision detection, not drift. What must not change is the speed.
    whole, split = Table(aim=0.0), Table(aim=0.0)
    for side in (whole, split):          # not `t` - `t` is the table the rest
                                         # of smoke() is still asserting against
        side.ball_x, side.ball_y, side.ball_vx, side.ball_vy = 640.0, 400.0, BALL_VX, 50.0
    for _ in range(2):                      # 2 thoughts: no face, no wall
        whole.advance()
        for _ in range(16):
            split.advance(1.0 / 16)
    assert whole.score == {"fish": 0, "visitor": 0}, "the free-flight leg hit something"
    assert abs(whole.ball_x - split.ball_x) < 1e-6, (whole.ball_x, split.ball_x)
    assert abs(whole.ball_y - split.ball_y) < 1e-6, (whole.ball_y, split.ball_y)
    assert int(whole.run_score) == int(split.run_score), "the score is not frac-linear"

    # -- the dino score: ticks, jumps on a return, resets on a miss -------
    t2 = Table(aim=0.0)
    # vy pinned: _serve picks it at random, and a random vy decides whether the
    # ball is still in front of the bat by the time it arrives
    t2.ball_x, t2.ball_y, t2.ball_vx, t2.ball_vy, t2.fish_y = 200.0, 400.0, -BALL_VX, 0.0, 400.0
    before = t2.run_score
    assert t2.advance() == "return_fish", "the ball should have come back off the paddle"
    assert t2.run_score >= before + SCORE_PER_RETURN, "a return has to pay"
    t2.run_score = 5000.0
    t2.ball_x, t2.ball_vx, t2.ball_vy = 200.0, -BALL_VX, 0.0
    t2.fish_y, t2.ball_y = PADDLE_H / 2, VIEW_H - BALL_R    # bat high, ball low
    assert t2.advance() == "point_visitor", "that should have been a miss"
    assert t2.run_score == 0.0, "a miss has to cost the run"
    # the losing thought still scores before the miss lands, so the record is
    # the run plus that thought's tick - the point is that it survived the reset
    assert t2.hi_score == 5000 + SCORE_PER_THOUGHT, t2.hi_score
    assert f"{t2.hi_score:05d}" in t2.ring()[0]["line"], t2.ring()[0]["line"]
    assert t2.ring()[0]["best"] is True, "first point over 0 has to be a best"

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
