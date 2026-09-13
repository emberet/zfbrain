#!/usr/bin/env python
"""Step 0: can a drawn table actually steer this brain?

Everything in the table-tennis feature rests on one number: `decode()["dy"]`,
the up/down DSGC asymmetry, is the only vertical control signal the fish has
(nMLF and vSPN measure 0.1-0.5 Hz on a web page, so a paddle driven by the bout
or turn channel would never move). This file measures it before a line of the
feature is written, and prints the numbers rather than asserting a hope.

Four probes:

  0a  retina only, no sim - deterministic. Is the 218 px / thought figure right?
  0b  full sim - the actual gate. Does |dy| separate moving from still?
  0c  closed loop - does a paddle driven by dy track a scripted ball?
  0d  the assumptions the plasticity design leans on, asserted cheaply.

The arithmetic 0a checks: Retina._downsample point-samples a 1280x800 frame at
rows linspace(0, 799, 12) and cols linspace(0, 1279, 18), and _motion is a
displaced-frame difference at a lag of exactly flow_shift=3 cells. So the
matched filter peaks when the image translates by 3 sample rows = 218 px
between retina.step() calls, at which point `up` is *exactly* zero and `down`
is the difference between samples 6 rows apart.

    python tools/probe_flow.py            # everything (loads the graph, ~1 min)
    python tools/probe_flow.py --retina   # 0a only, no graph, instant
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fishsim import FishSim, load_graph  # noqa: E402
from retina import Retina  # noqa: E402

VIEW_W, VIEW_H = 1280, 800
ROWS = np.linspace(0, VIEW_H - 1, 12).astype(int)
COLS = np.linspace(0, VIEW_W - 1, 18).astype(int)
DY_PX = int(ROWS[4] - ROWS[1])          # 3 sample rows = 218 px
DX_PX = int(COLS[4] - COLS[1])          # 3 sample cols = 225 px
# 6 sample rows apart must be *antiphase* or the `down` channel nulls too:
# 436 px = 1.5 periods -> period 290.7. Sinusoid, not a square wave: the
# retina point-samples and a square wave would alias.
GRATING_P = 2.0 * (ROWS[7] - ROWS[1]) / 3.0
MOTION_GAIN = 1.5                       # what roam.py:657 passes
BG = 232                                # site background grey, as talk.py used

_Y = np.arange(VIEW_H, dtype=np.float32)[:, None]


def grating(phase):
    """Full-field horizontal grating, drifting downward as `phase` grows."""
    band = 40.0 + 175.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (_Y - phase) / GRATING_P))
    return np.repeat(band.astype(np.uint8), VIEW_W, axis=1)


def ball_frame(cx, cy, r, bg=BG, fg=0, grating_phase=None):
    """A disc on a plain field - the naive table, and the thing most likely to
    be invisible: below ~75 px it falls between sample columns entirely."""
    if grating_phase is None:
        img = Image.new("L", (VIEW_W, VIEW_H), bg)
    else:
        img = Image.fromarray(grating(grating_phase), "L")
    d = ImageDraw.Draw(img)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    return np.asarray(img)


def _stats(rates):
    lf, rt = rates["dsgc_left"], rates["dsgc_right"]
    up, dn = rates["dsgc_up"], rates["dsgc_down"]
    return dict(up=up.mean(), down=dn.mean(), left=lf.mean(), right=rt.mean(),
                pinned=float((dn >= 24.99).mean()))


# ---------------------------------------------------------------------------
# 0a - retina only
# ---------------------------------------------------------------------------
def probe_a():
    print("=" * 72)
    print("0a  retina only: is 218 px/step the matched-filter peak?")
    print(f"    row samples {list(ROWS)}")
    print(f"    3 rows = {DY_PX} px   3 cols = {DX_PX} px   grating period {GRATING_P:.1f} px")
    print()
    print(f"    {'drift px':>9}  {'down Hz':>8} {'up Hz':>7} {'dy(drive)':>10} {'%pinned':>8}")
    # `down` saturates at every drift - it is clipped at motion_max and a
    # grating is high contrast everywhere. The discriminator is the *null*:
    # `up` collapses only when the image translates by exactly the 3-cell lag,
    # so the statistic that matters is the asymmetry, not the peak.
    peak, best, still = None, -1.0, None
    for drift in (0, 109, 150, 218, 290, 436):
        ret = Retina()
        s = None
        for k in range(6):
            out = ret.step(grating(k * drift))
            s = _stats(ret.rates(out, motion_gain=MOTION_GAIN))
        dy = (s["down"] - s["up"]) / max(s["down"] + s["up"], 1e-9)
        print(f"    {drift:>9}  {s['down']:>8.2f} {s['up']:>7.2f} {dy:>10.3f} {s['pinned']*100:>7.0f}%")
        if drift == 0:
            still = abs(dy)
        elif dy > best:
            best, peak = dy, drift
    print()
    print(f"    strongest asymmetry at {peak} px/step (predicted {DY_PX}), dy(drive) {best:.3f}")
    print(f"    a still grating is balanced, not silent: dy(drive) {still:.3f} "
          f"- the correlator has no motion gate, which is why a textured page "
          f"reads ~20 Hz on all four channels and steers nowhere")
    ok = peak == DY_PX and best > 0.5 and still < 0.02
    print(f"    0a {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 0b - full sim
# ---------------------------------------------------------------------------
def _frames(sim, groups, ret, frames, label, settle=0):
    """Feed `frames` through the retina into the brain exactly as roam.step
    does, and report decode() on each."""
    rows = []
    for i, fr in enumerate(frames):
        out = ret.step(fr)
        for g, hz in ret.rates(out, motion_gain=MOTION_GAIN).items():
            sim.set_drive(groups.get(g, []), hz)
        detail = sim.run(400)
        r = detail["rates_hz"]
        dec = sim.decode(r)
        if i >= settle:
            rows.append((r, dec))
    if not rows:
        return {}
    dys = [d["dy"] for _, d in rows]
    hot = max(max(v for v in r.values()) for r, _ in rows)
    esc = sum(1 for _, d in rows if d["escape"])
    scr = max(d["scroll"] for _, d in rows)
    r0 = rows[-1][0]
    print(f"    {label:<22} dy {np.mean(dys):+.3f} (|dy| {np.mean(np.abs(dys)):.3f}, "
          f"max {max(np.abs(dys)):.3f})  up {r0['dsgc_up']:.1f} dn {r0['dsgc_down']:.1f} "
          f"nmlf {r0['nmlf']:.1f}  scroll {scr:.2f}  escapes {esc}/{len(rows)}  hottest {hot:.0f} Hz")
    return dict(dy=float(np.mean(dys)), absdy=float(np.mean(np.abs(dys))),
                hot=float(hot), esc=esc, n=len(rows), scroll=float(scr))


def probe_b(graph_path, groups_path):
    print("=" * 72)
    print("0b  full sim: does a drawn stimulus separate moving from still?")
    t0 = time.time()
    graph = load_graph(graph_path, groups_path)
    groups = graph["groups"]
    print(f"    graph: {graph['meta'].get('label')} "
          f"{graph['meta'].get('neurons'):,} neurons, loaded in {time.time()-t0:.1f}s")
    print()

    def fresh():
        sim = FishSim(graph)
        ret = Retina()
        # settle on a still field first, the way smoke() does
        for _ in range(8):
            out = ret.step(grating(0.0))
            for g, hz in ret.rates(out, motion_gain=MOTION_GAIN).items():
                sim.set_drive(groups.get(g, []), hz)
            sim.run(400)
        return sim, ret

    res = {}
    sim, ret = fresh()
    res["still grating"] = _frames(sim, groups, ret, [grating(0.0)] * 8, "still grating")

    sim, ret = fresh()
    res["drift grating"] = _frames(
        sim, groups, ret, [grating(k * DY_PX) for k in range(1, 13)], "grating +218 px/f")

    sim, ret = fresh()
    res["drift grating up"] = _frames(
        sim, groups, ret, [grating(-k * DY_PX) for k in range(1, 13)], "grating -218 px/f")

    # the ball on a plain field: the dilution case. A few of 216 grid cells
    # against a population mean over 26,730 DSGC neurons.
    for r in (40, 90, 150):
        sim, ret = fresh()
        ys = [ROWS[1] + (k % 4) * DY_PX for k in range(13)]
        frames = [ball_frame(640, y, r) for y in ys[1:]]
        res[f"ball r={r}"] = _frames(sim, groups, ret, frames, f"ball r={r} on plain")

    # the ball *plus* a drifting grating - the OMR rig with a scoreboard
    sim, ret = fresh()
    frames = [ball_frame(200 + k * DX_PX, ROWS[4], 90, grating_phase=k * DY_PX)
              for k in range(1, 6)] * 3
    res["ball + grating"] = _frames(sim, groups, ret, frames, "ball + grating")

    print()
    still, drift = res["still grating"], res["drift grating"]
    up = res["drift grating up"]
    checks = [
        ("|dy| >= 0.08 on a drifting grating", drift["absdy"] >= 0.08),
        ("|dy| <= 0.02 on a still field", still["absdy"] <= 0.02),
        ("drift sign follows drift direction", drift["dy"] > 0 > up["dy"]),
        ("nothing saturates (< 350 Hz)", max(v["hot"] for v in res.values()) < 350.0),
        ("escape quiet while drifting", drift["esc"] <= 2),
    ]
    for name, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    note: nmlf bout on a coherent grating -> scroll {drift['scroll']:.2f} "
          f"({'bouts' if drift['scroll'] > 0.3 else 'no bout'})")
    return all(ok for _, ok in checks), res, graph


# ---------------------------------------------------------------------------
# 0c - closed loop
# ---------------------------------------------------------------------------
def probe_c(graph, ticks=40, gain=DY_PX, use_grating=True):
    print("=" * 72)
    print(f"0c  closed loop: {ticks} thoughts, paddle driven by dy "
          f"({'grating' if use_grating else 'raw ball'})")
    groups = graph["groups"]
    sim, ret = FishSim(graph), Retina()
    y_lo, y_hi = ROWS[1], ROWS[10]
    ball_y, ball_vy = float(ROWS[4]), float(DY_PX)
    ball_x, ball_vx = 200.0, float(DX_PX)
    paddle = float(VIEW_H / 2)
    phase = 0.0
    trace, pinned, revs = [], 0, []
    prev_sign = 0
    for k in range(ticks):
        err = ball_y - paddle
        if use_grating:
            phase += DY_PX * (1 if err > 40 else (-1 if err < -40 else 0))
            fr = ball_frame(int(ball_x), int(ball_y), 90, grating_phase=phase)
        else:
            fr = ball_frame(int(ball_x), int(ball_y), 90)
        out = ret.step(fr)
        for g, hz in ret.rates(out, motion_gain=MOTION_GAIN).items():
            sim.set_drive(groups.get(g, []), hz)
        dec = sim.decode(sim.run(400)["rates_hz"])
        paddle = min(y_hi, max(y_lo, paddle + dec["dy"] * gain))
        if paddle in (y_lo, y_hi):
            pinned += 1
        trace.append((ball_y, paddle, dec["dy"]))
        s = 1 if ball_vy > 0 else -1
        if s != prev_sign:
            revs.append(k)
            prev_sign = s
        ball_y += ball_vy
        if ball_y >= y_hi or ball_y <= y_lo:
            ball_vy = -ball_vy
            ball_y = min(y_hi, max(y_lo, ball_y))
        ball_x += ball_vx
        if ball_x >= VIEW_W - 100 or ball_x <= 100:
            ball_vx = -ball_vx
    err = np.mean([abs(b - p) for b, p, _ in trace])
    moved = np.mean([abs(d) for _, _, d in trace])
    print("    tick  ball_y  paddle    dy")
    for i, (b, p, d) in enumerate(trace):
        if i % max(1, ticks // 20) == 0:
            print(f"    {i:>4}  {b:>6.0f}  {p:>6.0f}  {d:>+.3f}")
    print(f"    mean |ball-paddle| {err:.0f} px   mean |dy| {moved:.3f}   "
          f"pinned {pinned/ticks*100:.0f}% of ticks")
    ok = moved > 0.05 and pinned / ticks < 0.30
    print(f"    0c {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 0d - assumptions the plasticity design leans on
# ---------------------------------------------------------------------------
def probe_d(graph):
    print("=" * 72)
    print("0d  assumptions")
    sim = FishSim(graph)
    g = graph["groups"]
    ret_idx = np.asarray(g["retina"], dtype=np.int64)
    n_ret = len(ret_idx)
    facts = []
    facts.append(("weights are writeable in memory", bool(sim.weights.flags.writeable)))
    facts.append((f"retina is arange(0, {n_ret})",
                  bool(np.array_equal(np.sort(ret_idx), np.arange(n_ret)))))
    hi = int(sim.indptr[n_ret])
    facts.append((f"retina outgoing block is contiguous [0, {hi:,})", hi > 0))
    blk = np.abs(sim.weights[:hi])
    uniq = np.unique(blk)
    facts.append((f"retina block has one magnitude ({uniq[0]:.8f})", uniq.size == 1))
    dsgc = np.concatenate([np.asarray(g[f"dsgc_{d}"], dtype=np.int64)
                           for d in ("up", "down", "left", "right")])
    facts.append(("retina projects only to DSGC",
                  bool(np.isin(sim.targets[:hi], dsgc).all())))
    lo, dhi = int(dsgc.min()), int(dsgc.max())
    facts.append((f"dsgc is contiguous [{lo:,}, {dhi+1:,})",
                  bool(np.array_equal(np.sort(dsgc), np.arange(lo, dhi + 1)))))
    for name, ok in facts:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    scale {sim.weight_scale:.6f}   edges {len(sim.weights):,}   "
          f"plastic outgoing {int(sim.indptr[dhi+1]):,}")
    return all(ok for _, ok in facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="build/graph.npz")
    ap.add_argument("--groups", default="build/groups.json")
    ap.add_argument("--retina", action="store_true", help="0a only, no graph")
    ap.add_argument("--ticks", type=int, default=40)
    args = ap.parse_args()
    ok = probe_a()
    if args.retina:
        return
    ok_b, res, graph = probe_b(args.graph, args.groups)
    ok_d = probe_d(graph)
    ok_c = probe_c(graph, ticks=args.ticks)
    print("=" * 72)
    print(f"0a {'PASS' if ok else 'FAIL'}   0b {'PASS' if ok_b else 'FAIL'}   "
          f"0c {'PASS' if ok_c else 'FAIL'}   0d {'PASS' if ok_d else 'FAIL'}")


if __name__ == "__main__":
    main()
