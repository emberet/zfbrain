"""A map from brain state onto a fixed word list. No model on any path.

The fish still has no language. What it has is a fixed list of 850 words, a
12-channel reading of its own firing, and a map between them that is its own and
changes with what it has seen. It does not know what any of the words mean.

Two halves, and the page says which is which:

  * THE ANCHORED HALF is ours. `ANCHORS` below is a hand-written table from six
    measurable axes onto 58 words, and an axis contributes a word only when it
    is past a stated z-threshold. Within an axis the word is picked by
    magnitude, so `quick` and `danger` are the same channel an octave apart.
    These words mean what we decided they mean. We wrote them, and the site
    prints this table verbatim so you can see us writing them.

  * THE GROWN HALF is not. The remaining 792 words are labels stapled onto
    clusters of its own firing, found by an unsupervised quantiser that starts
    at one prototype and splits a cell only when the cell has seen SPLIT_N
    thoughts AND its spread along its own principal axis exceeds a noise floor
    measured on a held frame (tools/probe_lexicon.py, gate 0a). A word here is
    arbitrary by construction and carries no meaning at all; what carries
    information is *which* word, consistently, for the same state.

Why grown and not a fixed 850 prototypes: Lloyd-style VQ allocates codewords as
density^(d/(d+2)). The fish spends most of its life looking at a static text
page, which is one dominant blob, so a fixed codebook packs nearly every
prototype inside that blob and splits it below the per-thought noise floor.
Every word gets used, the histogram looks rich, and "same state -> same word"
silently fails. A dead vocabulary is honest. A live-but-random one falsifies the
only claim this file makes, while looking like success.

Why the slow channels are high-passed: `habituation` and `intrinsic` are slow
monotone drifts with far more range than the nMLF/vSPN deviations they would sit
next to. Raw, they dominate the distance metric and the word becomes a clock —
time since the page changed, wearing a vocabulary. They enter as x - EMA_slow(x)
only, and gate 0e regresses every channel against elapsed time and drops the
ones time explains.

Nothing here touches the connectome. `features()` is a read of numbers the brain
already publishes; the codebook is a small matrix outside it. No weights, no
drive, no calibration, and no reward signal: a word is not scored, so there is
nothing here for the fish to get better at.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np

WORDS_PATH = Path(__file__).with_name("data") / "basic_english.txt"
LIST_SIZE = 850           # asserted against the file on load

# -- the feature vector --------------------------------------------------
# 12 channels, all read from stream()/decode(). `scroll` and `turn` are monotone
# in nmlf/vspn and excluded as redundant; the four DSGC directions collapse to
# one magnitude plus the two steering terms decode() already computes; `escape`
# is too rare for a Euclidean term and is an anchor trigger instead.
FEATURES = ("dx", "dy", "flow", "retina", "nmlf", "vspn", "mauthner",
            "spinal", "membrane", "spikes", "hab_hp", "theta_hp")
NDIM = len(FEATURES)

# decode()'s own span rule, reused verbatim so a channel here rises exactly when
# the behaviour readout says it rises (fishsim.py:785).
MIN_SPAN = 8.0
SPAN_FACTOR = 1.5

CLIP = 4.0                # z-clip: one Mauthner burst may not drag a prototype
NORM_TAU = 1000.0         # thoughts; running mean/var for the z-score
HP_TAU = 200.0            # thoughts; the high-pass corner for the slow channels
WARMUP = 200              # thoughts of silence while the z-score finds its
                          # scale. Every threshold in the anchor table is stated
                          # in standard deviations of this fish's own range, so
                          # before there is an estimate of that range there is
                          # no meaning in comparing anything to 1.5. ~3.5 min of
                          # a life, and the site says so rather than hiding it.

# -- the grown codebook --------------------------------------------------
SPLIT_N = 400             # a cell must have seen this many thoughts to split
NOISE_FLOOR = 0.35        # measured, not chosen: probe_lexicon.py gate 0a
ETA_MIN = 0.01            # floor on the MacQueen rate 1/n_i, disclosed
SAVE_EVERY_S = 300.0

# -- the anchored table --------------------------------------------------
# axis -> (channel(s), threshold in z, words weakest -> strongest).
# An axis fires only past its threshold; the word inside it is chosen by
# magnitude, one word per Z_STEP of z beyond the threshold.
Z_STEP = 1.0

ANCHORS = {
    # a Mauthner burst is the one thing in this brain that is all-or-nothing
    "startle+":    (("mauthner",), 2.0,
                    ("quick", "fear", "run", "sudden", "shock", "violent",
                     "danger")),
    "startle-":    (("mauthner",), -1.5, ("quiet", "safe", "soft", "peace")),
    # short-term depression rising = it has been looking at this a while
    "familiar+":   (("hab_hp",), 1.5,
                    ("again", "same", "common", "regular", "old")),
    "familiar-":   (("hab_hp",), -1.5,
                    ("different", "new", "strange", "first")),
    # intrinsic threshold rising = the cells have been working
    "fatigue+":    (("theta_hp",), 1.5,
                    ("slow", "rest", "tired", "feeble", "sleep")),
    "fatigue-":    (("theta_hp",), -1.5, ("awake", "ready", "healthy")),
    "arousal+":    (("spikes", "membrane"), 1.5,
                    ("much", "bright", "strong", "loud", "full")),
    "arousal-":    (("spikes", "membrane"), -1.5,
                    ("little", "low", "small", "thin", "dark")),
    "drive+":      (("spinal",), 1.5,
                    ("move", "go", "forward", "start", "push")),
    "drive-":      (("spinal",), -1.5, ("waiting", "stop", "still")),
    "flow+":       (("flow",), 1.5, ("across", "wave", "current", "round")),
    "flow-":       (("flow",), -1.5, ("fixed", "level", "flat", "straight")),
}

# Orientation is a direction, not a magnitude, so it gets its own rule — but it
# is stated in the same z as every other axis, not in decode()'s raw -1..1. The
# raw threshold was 0.3 and on 58 consecutive thoughts of real browsing dy had a
# standard deviation of 0.003: 0.3 is a hundred sigma out, so "up" and "down"
# could not fire on a page. They were not rare, they were unreachable. Reading
# them in sigma makes the word mean "drifting further than this fish usually
# does", which is what every other word here already means.
#
# `left`/`right` are struck for the same reason hab_hp and theta_hp are: gate
# 0b measured dx's across-condition spread at 1.36x its own noise, under the 2x
# bar, so dx does not tell conditions apart. dy passed at 2.47x and stays.
ORIENT = {"up": ("dy", -1), "down": ("dy", 1),
          "left": ("dx", -1), "right": ("dx", 1)}
ORIENT_T = 1.5            # in z, like the rest of the table
DEAD_ORIENT = ("left", "right")   # dx: 0b span 1.36x noise, under the 2x bar

MAX_WORDS = 4             # what utter() will ever return

# Channels that gate 0e found are clocks rather than states, and that gate 0b
# found do not separate conditions at all. Measured, not suspected:
#
#     hab_hp     R^2 vs elapsed time 0.500   vs condition 0.023   span 0.16
#     theta_hp   R^2 vs elapsed time 0.693   vs condition 0.015   span 0.13
#
# High-passing them was supposed to be enough and was not. They stay in
# FEATURES because that is what the probe measured and the published numbers
# have to describe the thing that was measured — but no word is allowed to come
# out of them, because "familiar" and "tired" driven by a channel that elapsed
# time predicts forty times better than the stimulus does would be a clock
# wearing a vocabulary, which is exactly the failure this file exists to avoid.
# The page prints these two axes struck through, with these numbers.
CLOCK_CHANNELS = ("hab_hp", "theta_hp")


def axis_live(axis):
    """False for an axis whose channels time explains better than the world
    does. Such an axis is kept in ANCHORS, printed, and never uttered."""
    chans, _t, _w = ANCHORS[axis]
    return not any(c in CLOCK_CHANNELS for c in chans)


def anchored_words():
    """Every word the hand-written table can actually produce — the struck-out
    axes are not in it, so the counts on the page are the live ones."""
    out = []
    for axis, (_chan, _t, words) in ANCHORS.items():
        if axis_live(axis):
            out.extend(words)
    out.extend(w for w in ORIENT if w not in DEAD_ORIENT)
    return out


def reserved_words():
    """Every word the table names, live axis or not. The codebook may not have
    any of these — see Lexicon.__init__."""
    out = []
    for _chan, _t, words in ANCHORS.values():
        out.extend(words)
    out.extend(ORIENT)
    return out


def load_words(path=WORDS_PATH):
    """The word list, asserted. This file is the moderation policy: strangers
    compose from the same list, so nothing free-text ever reaches a renderer."""
    lines = Path(path).read_text().splitlines()
    words = [w.strip() for w in lines if w.strip() and not w.startswith("#")]
    if len(words) != LIST_SIZE:
        raise ValueError(f"{path}: {len(words)} words, expected {LIST_SIZE}")
    if len(set(words)) != len(words):
        raise ValueError(f"{path}: duplicate words")
    missing = [w for w in anchored_words() if w not in set(words)]
    if missing:
        raise ValueError(f"anchors missing from {path}: {missing}")
    return words


def _span(hz, rest_hz):
    """decode()'s span rule: a rise above the fish's own recent baseline,
    scaled by that baseline, floored so a quiet population is not all noise."""
    return (hz - rest_hz) / max(MIN_SPAN, SPAN_FACTOR * rest_hz)


def features(detail, dec, rest):
    """(12,) float32 from one thought. Pure — no state, no clock, no RNG.

    The two slow channels come out raw here; the high-pass that keeps them from
    becoming a clock lives in Lexicon.encode(), because it needs memory."""
    r = detail.get("rates_hz", {})
    rest = rest or {}
    up = r.get("dsgc_up", 0.0) + r.get("dsgc_down", 0.0)
    lr = r.get("dsgc_left", 0.0) + r.get("dsgc_right", 0.0)
    v = np.empty(NDIM, dtype=np.float32)
    v[0] = dec.get("dx", 0.0)
    v[1] = dec.get("dy", 0.0)
    v[2] = np.log1p(up + lr)
    for i, g in enumerate(("retina", "nmlf", "vspn", "mauthner", "spinal"), 3):
        v[i] = _span(r.get(g, 0.0), rest.get(g, 0.0))
    v[8] = detail.get("mean_membrane_mv", 0.0)
    v[9] = np.log1p(max(0.0, float(detail.get("spikes_per_sec", 0.0))))
    v[10] = detail.get("habituation", 0.0)
    v[11] = detail.get("intrinsic", 0.0)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


class Lexicon:
    """The map. `observe` learns, `quantise` does not, and both are read-only on
    everything the brain owns."""

    HP_CHANNELS = (10, 11)   # hab_hp, theta_hp

    def __init__(self, path=".state/lexicon.npz", words=None, learn=True):
        self.path = Path(path)
        self.words = list(words) if words else load_words()
        self.learn = learn
        # The grown half never reaches for a word the table owns — and that
        # includes the struck-out axes. A word we wrote a meaning for does not
        # become free to relabel just because the channel behind it turned out
        # to be a clock; handing "tired" to an arbitrary cluster would be worse
        # than leaving it unused.
        free = [w for w in self.words if w not in set(reserved_words())]
        # Deterministic and visibly arbitrary. Ogden's list is alphabetical, and
        # walking it in order would hand the first clusters "a", "able",
        # "about" — words that read as grammar and would imply a meaning that is
        # not there. A fixed-seed shuffle is just as reproducible (seed 0, this
        # line, forever) and does not dress an arbitrary label up as sense.
        random.Random(0).shuffle(free)
        self.free = free

        self.mean = np.zeros(NDIM, dtype=np.float64)
        self.var = np.ones(NDIM, dtype=np.float64)
        self.slow = np.zeros(NDIM, dtype=np.float64)   # for the high-pass
        self.seen = 0
        self.proto = np.zeros((0, NDIM), dtype=np.float32)
        self.count = np.zeros(0, dtype=np.int64)       # MacQueen n_i
        self.total = np.zeros(0, dtype=np.int64)       # lifetime visits
        self.scatter = np.zeros((0, NDIM, NDIM), dtype=np.float64)
        self.cell_word = []                            # index -> word
        self.splits = 0
        self.last = None                               # (idx, word)
        self.last_anchors = []
        # start the clock now, not at the epoch: a 0.0 here means the very
        # first observe() tries to write the file, and every observe() after it
        # until one succeeds — which on a read-only path is a disk call per
        # thought, and showed up as 0.7 ms in probe 0f.
        self._saved_at = time.time()
        # the codebook can never outgrow the words left for it: every word the
        # table names is reserved, struck-out axes included
        self._max_cells = len(self.free)
        self.load()

    # -- normalisation ---------------------------------------------------
    def encode(self, vec):
        """Raw features -> the working vector: high-pass the two slow channels,
        z-score everything against a running mean/var, clip to +/-CLIP.

        The averaging rate is `1/seen` until the window is full and `1/NORM_TAU`
        after, which makes the first NORM_TAU thoughts an exact sample mean and
        variance rather than a decayed guess. That is not a refinement; without
        it the whole anchored half is mute. `self.var` used to start at 1.0 and
        decay at 1/1000 a thought, so after n thoughts it still carried
        0.999**n of that 1.0 — and every real channel here has a standard
        deviation far below one:

            mauthner 0.24   spikes 0.23   membrane 0.17   spinal 0.16
            flow 0.055      vspn 0.035    dx 0.002

        Measured on 58 consecutive thoughts of the live fish, every channel's
        variance estimate read 0.945 (= 0.999**58) instead of its own, so every
        z was divided by ~1.0 instead of ~0.2 and the largest |z| in the whole
        run was 0.675 against a 1.5 threshold. Nothing could ever fire. The
        prior would have taken ~2,900 thoughts to decay far enough for
        `mauthner` and ~12,000 for `dx`, staggered per channel, and with the
        codebook frozen none of it was persisted — so each life restarted the
        wait. The live fish said nothing at all for 338 thoughts, which is how
        this was found."""
        v = np.asarray(vec, dtype=np.float64).copy()
        a_slow = 1.0 / HP_TAU
        for i in self.HP_CHANNELS:
            self.slow[i] += a_slow * (v[i] - self.slow[i])
            v[i] -= self.slow[i]
        if self.seen == 0:
            self.mean[:] = v
            self.var[:] = 0.0
        else:
            a = 1.0 / min(float(self.seen), NORM_TAU)
            d = v - self.mean
            self.mean += a * d
            # Welford's product, d_old * d_new: with a = 1/n this is exactly the
            # sample variance, and it converges from 0 instead of from a prior.
            self.var += a * (d * (v - self.mean) - self.var)
        self.seen += 1
        if self.seen < WARMUP:
            # Below WARMUP the variance is estimated off a handful of samples
            # and a z of 8 means "third thought of this life", not "unusual".
            # Saying nothing is the honest output of not knowing the scale yet.
            return np.zeros(NDIM, dtype=np.float32)
        z = (v - self.mean) / np.sqrt(np.maximum(self.var, 1e-9))
        return np.clip(z, -CLIP, CLIP).astype(np.float32)

    # -- the codebook ----------------------------------------------------
    def _nearest(self, z):
        if len(self.proto) == 0:
            return -1, float("inf")
        d = np.linalg.norm(self.proto - z, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def _add(self, centre):
        self.proto = np.vstack([self.proto, centre.astype(np.float32)])
        self.count = np.append(self.count, 1)
        self.total = np.append(self.total, 0)
        self.scatter = np.concatenate(
            [self.scatter, np.zeros((1, NDIM, NDIM))], axis=0)
        w = self.free[len(self.cell_word) % len(self.free)]
        self.cell_word.append(w)
        return len(self.proto) - 1

    def _try_split(self, i):
        """Split cell i only if it has seen enough AND is genuinely wider than
        the measured noise floor along its own principal axis. Both conditions,
        because either alone splits noise."""
        if self.count[i] < SPLIT_N or len(self.proto) >= self._max_cells:
            return False
        cov = self.scatter[i] / max(1, self.count[i])
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return False
        lam = float(vals[-1])
        if lam <= 0 or np.sqrt(lam) <= NOISE_FLOOR:
            return False
        axis = vecs[:, -1] * (0.5 * np.sqrt(lam))
        centre = self.proto[i].astype(np.float64)
        j = self._add(centre + axis)
        self.proto[i] = (centre - axis).astype(np.float32)
        # both children restart their MacQueen counters, so neither inherits a
        # rate so slow it can never move again
        self.count[i] = 1
        self.count[j] = 1
        self.scatter[i] = 0.0
        self.scatter[j] = 0.0
        self.splits += 1
        return True

    def think(self, vec, dec):
        """One thought, start to finish. The only entry point roam.py uses.

        Exactly one `encode()` per thought, so the running statistics advance
        once and only once — and they advance whether or not we are learning.
        That distinction cost a restart to find. The z-score is not learning:
        it is the scale the hand-written thresholds are *stated in*, so with it
        frozen at mean 0 / var 1 every threshold is being compared against a
        range the brain never lives on. With ZF_LEXICON_LEARN off, `observe()`
        was never called, `seen` stayed 0, and the live fish said `current` and
        nothing else on every thought for a whole life."""
        z = self.encode(vec)
        if self.learn:
            self._learn(z)
        # Saved either way, because the running mean/var is not the codebook.
        # With learning off the prototypes go back to disk byte-identical to
        # the ones that came off it — what actually changes is the scale, and a
        # scale that is thrown away at every restart makes the fish mute for
        # the first WARMUP thoughts of every life and gives the same brain
        # state a different word either side of a crash.
        self.maybe_save()
        return self._say(z, dec)

    def observe(self, vec):
        """One thought through the codebook alone. Kept for
        tools/probe_lexicon.py, which drives the map without a roamer."""
        z = self.encode(vec)
        return self._learn(z)

    def _learn(self, z):
        """The codebook update. Caller has already encoded."""
        if len(self.proto) == 0:
            i = self._add(z.astype(np.float64))
            self.total[i] += 1
            self.last = (i, self.cell_word[i])
            return i, self.cell_word[i], 0.0, True
        i, dist = self._nearest(z)
        self.total[i] += 1
        split = False
        if self.learn:
            self.count[i] += 1
            r = z.astype(np.float64) - self.proto[i]
            self.scatter[i] += np.outer(r, r)
            eta = max(ETA_MIN, 1.0 / float(self.count[i]))
            self.proto[i] = (self.proto[i] + eta * r).astype(np.float32)
            split = self._try_split(i)
        self.last = (i, self.cell_word[i])
        return i, self.cell_word[i], dist, split

    def quantise(self, vec):
        """Frozen: the same read, with nothing updated — not the prototypes and
        not the running statistics. This is how a stability claim gets made: the
        map cannot move while it is being tested, and utter() can call it
        alongside anchors() without counting one thought twice."""
        return self._quantise_z(self.encode_readonly(vec))

    def _quantise_z(self, z):
        i, _d = self._nearest(z)
        if i < 0:
            return -1, None
        return i, self.cell_word[i]

    # -- the hand-written half -------------------------------------------
    def anchors(self, vec, dec):
        """The words we wrote, for the state we wrote them for. Returns a list
        of (word, axis, z) so the site can print why each one fired."""
        return self._anchors_z(self.encode_readonly(vec), dec)

    def _anchors_z(self, z, dec):
        out = []
        for axis, (chans, thresh, words) in ANCHORS.items():
            if not axis_live(axis):
                continue          # 0e: time explains this channel, not the world
            val = float(np.mean([z[FEATURES.index(c)] for c in chans]))
            if thresh >= 0:
                if val < thresh:
                    continue
                over = val - thresh
            else:
                if val > thresh:
                    continue
                over = thresh - val
            k = min(len(words) - 1, int(over / Z_STEP))
            out.append((words[k], axis, round(val, 2)))
        if dec and dec.get("escape"):
            out.insert(0, ("danger", "escape", 1.0))
        for word, (chan, sign) in ORIENT.items():
            if word in DEAD_ORIENT:
                continue      # 0b: dx does not separate conditions
            v = float(z[FEATURES.index(chan)])
            if sign * v > ORIENT_T:
                out.append((word, "orientation", round(v, 2)))
        return out

    def encode_readonly(self, vec):
        """encode() without advancing the running statistics, so calling
        anchors() and observe() on the same thought does not double-count."""
        if self.seen < WARMUP:
            return np.zeros(NDIM, dtype=np.float32)
        v = np.asarray(vec, dtype=np.float64).copy()
        for i in self.HP_CHANNELS:
            v[i] -= self.slow[i]
        z = (v - self.mean) / np.sqrt(np.maximum(self.var, 1e-9))
        return np.clip(z, -CLIP, CLIP)

    # -- what it says ------------------------------------------------------
    def utter(self, vec, dec):
        """At most MAX_WORDS: the anchors that fired, strongest first, then the
        grown word for this state. Not a sentence — there is no grammar here and
        the fish is not composing one."""
        return self._say(self.encode_readonly(vec), dec)

    def _say(self, z, dec):
        anc = self._anchors_z(z, dec)
        anc.sort(key=lambda t: -abs(t[2]))
        words, seen = [], set()
        for w, _axis, _z in anc[:MAX_WORDS - 1]:
            if w not in seen:
                words.append(w)
                seen.add(w)
        _i, w = self._quantise_z(z)
        if w and w not in seen:
            words.append(w)
        self.last_anchors = [a[0] for a in anc]
        return words[:MAX_WORDS]

    # -- persistence -------------------------------------------------------
    # Load-bearing and easy to miss: without the file the map resets every life
    # and "it changes with what it has seen" is not true of anything.
    def maybe_save(self):
        now = time.time()
        if now - self._saved_at >= SAVE_EVERY_S:
            self.save()

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # np.savez appends ".npz" to any name that does not already end in
            # it, so the temp file has to end in .npz or os.replace below looks
            # for a path that was never written.
            tmp = self.path.with_name(self.path.name + ".tmp.npz")
            np.savez(tmp, proto=self.proto, count=self.count, total=self.total,
                     scatter=self.scatter, mean=self.mean, var=self.var,
                     slow=self.slow, seen=np.int64(self.seen),
                     splits=np.int64(self.splits),
                     cell_word=np.array(self.cell_word, dtype=object),
                     version=np.int64(1))
            os.replace(tmp, self.path)
            self._saved_at = time.time()
        except OSError as exc:
            print(f"lexicon save: {exc}")

    def load(self):
        if not self.path.exists():
            return
        try:
            d = np.load(self.path, allow_pickle=True)
            if int(d["version"]) != 1 or d["proto"].shape[1] != NDIM:
                return
            self.proto = d["proto"].astype(np.float32)
            self.count = d["count"]
            self.total = d["total"]
            self.scatter = d["scatter"]
            self.mean, self.var, self.slow = d["mean"], d["var"], d["slow"]
            self.seen = int(d["seen"])
            self.splits = int(d["splits"])
            self.cell_word = [str(w) for w in d["cell_word"]]
        except (OSError, KeyError, ValueError, IndexError) as exc:
            print(f"lexicon load: {exc} — starting from one prototype")

    # -- the heartbeat -----------------------------------------------------
    def state(self):
        live = int((self.total > 0).sum())
        order = np.argsort(-self.total)[:8] if len(self.total) else []
        return {
            "list": len(self.words),
            "anchored": len(set(anchored_words())),      # words it can say
            "reserved": len(set(reserved_words())),       # words we named at all
            "cells": len(self.proto),
            "live_words": live,
            "thoughts": self.seen,
            "splits": self.splits,
            "learning": bool(self.learn),
            # Silent until the z-score has a scale, and says which.
            "warmup": WARMUP,
            "warm": bool(self.seen >= WARMUP),
            "word": self.last[1] if self.last else None,
            "anchors": list(self.last_anchors),
            "top": [{"word": self.cell_word[int(i)],
                     "visits": int(self.total[int(i)])} for i in order],
        }


def main():
    """`python lexicon.py` — the table and the list, printed. No brain needed."""
    words = load_words()
    anchored = anchored_words()
    reserved = reserved_words()
    print(f"list: {len(words)}  live anchors: {len(set(anchored))}  "
          f"reserved: {len(set(reserved))}  "
          f"grown ceiling: {len(words) - len(set(reserved))}")
    print(f"features ({NDIM}): {', '.join(FEATURES)}")
    for axis, (chans, t, ws) in ANCHORS.items():
        mark = "  " if axis_live(axis) else " *"
        print(f" {mark}{axis:<11} {'+'.join(chans):<18} z{t:+.1f}  {' '.join(ws)}")
    orient = " ".join(f"*{w}" if w in DEAD_ORIENT else w for w in ORIENT)
    print(f"   {'orientation':<11} {'dy (dx struck)':<18} "
          f"z>{ORIENT_T:+.1f}  {orient}")
    print(f"  * struck: {', '.join(CLOCK_CHANNELS)} are clocks, not states "
          f"(gate 0e); dx spans 1.36x its noise, under 0b's 2x bar. "
          f"Printed, never uttered.")
    print(f"  silent for the first {WARMUP} thoughts of a life, while the "
          f"z-score finds this fish's own range.")
    dupe = [w for w in set(reserved) if reserved.count(w) > 1]
    print("duplicate anchors:", dupe or "none")


if __name__ == "__main__":
    main()
