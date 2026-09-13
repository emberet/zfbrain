#!/usr/bin/env python
"""Play the table against the real brain, with no browser, and see if it learns.

Two jobs:

  1. exercise Roamer._rally() end to end on the real graph - no Playwright, no
     network - so the rally path is tested without waiting for a visitor.
  2. the falsifiable one. Run the same rallies with the Hebbian rule on and
     off and compare: mean |dy|, how close the paddle tracks the ball, and how
     far the retina's weight distribution actually moved.

There is no reward anywhere in this, so there is no reason to expect the fish
to get better at pong, and "no measurable difference" is a perfectly good
answer. Whatever this prints is what the site is allowed to say.

    .venv/bin/python tools/rally.py --rallies 8
    .venv/bin/python tools/rally.py --rallies 30 --compare
    .venv/bin/python tools/rally.py --rallies 50 --thoughts 40 --soak
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ZF_PONG", "1")

import fishsim  # noqa: E402
import pong  # noqa: E402
import roam  # noqa: E402


def build(seed=7):
    """A Roamer with everything the rally path needs and nothing else. __init__
    would start a chain-poll thread and build the 187k-neuron graph doc; this
    takes the same object without the parts a headless rally cannot use."""
    r = roam.Roamer.__new__(roam.Roamer)
    r.graph = fishsim.load_graph("build/graph.npz", "build/groups.json")
    r.meta = r.graph["meta"]
    r.sim = fishsim.FishSim(r.graph, seed=seed)
    r.retina = roam.Retina()
    r.groups = r.graph["groups"]
    r.stats = {"pages": 0, "clicks": 0, "scrolls": 0, "vetoed": 0,
               "escapes": 0, "brain_steps": 0, "points": 0}
    r.events = roam.deque(maxlen=12)
    r.feed = roam.Feed()
    r._seq = r._mask_seq = 0
    r._mask_at = r._cam_at = r._last_file_write = 0.0
    r._life = 0
    r._rest = {}
    r._cursor = (roam.VIEW_W // 2, roam.VIEW_H // 2)
    r._page_url = r._page_title = None
    r._cam_page = r._last = None
    r._chain = {"network": "mainnet", "slot": None, "at": 0.0, "mint": None,
                "supply": None, "sol": None, "bag": None}
    r.state_dir = ".state"
    r.pong = pong
    r.table = pong.Table()
    return r


def play(r, rallies, thoughts):
    """Returns a row per rally: mean |dy| and mean paddle-to-ball error."""
    roam.RALLY_THOUGHTS = thoughts
    out = []
    for i in range(rallies):
        ok, tok, _ = r.table.join(f"10.0.0.{i}")
        assert ok, "could not take the seat"
        r.table._per_ip_s = 0.0
        dys, errs = [], []
        # watch the rally from outside by wrapping the one call that moves the
        # fish's paddle - that is the only path from brain to motor, so this
        # cannot miss a contribution or count one that is not the fish's
        real_move = r.table.move_fish

        def spy(dy, _real=real_move):
            dys.append(abs(float(dy)))
            errs.append(abs(r.table.ball_y - r.table.fish_y))
            _real(dy)

        r.table.move_fish = spy
        try:
            r._rally()
        finally:
            r.table.move_fish = real_move
            r.table.release(tok)
        out.append((float(np.mean(dys)) if dys else 0.0,
                    float(np.mean(errs)) if errs else 0.0))
        print(f"    rally {i + 1:>2}: |dy| {out[-1][0]:.3f}   "
              f"paddle-ball {out[-1][1]:6.1f} px   score {r.table.score}")
    return np.array(out)


def retina_spread(sim):
    """How far the retina's outgoing weights have moved off their single
    starting magnitude. Zero before any learning, by construction: probe 0d
    checks that every one of those 6.8M synapses starts at 0.30084598."""
    h = sim._hebb
    if h is None:
        return 0.0
    return float(np.abs(sim.weights[:h["hi"]]).std())


def run_one(tag, rallies, thoughts, hebb):
    fishsim.set_plasticity(ip=1, dep2=1, sens=1, hebb=hebb)
    print(f"  {tag} (hebbian {'on' if hebb else 'off'})")
    r = build()
    rows = play(r, rallies, thoughts)
    spread = retina_spread(r.sim)
    half = max(1, len(rows) // 2)
    first, last = rows[:half], rows[half:]
    print(f"    mean |dy| {rows[:, 0].mean():.4f}   mean paddle-ball "
          f"{rows[:, 1].mean():.1f} px   retina |w| spread {spread:.2e}")
    print(f"    first half -> last half:  |dy| {first[:, 0].mean():.4f} -> "
          f"{last[:, 0].mean():.4f}   error {first[:, 1].mean():.1f} -> "
          f"{last[:, 1].mean():.1f} px")
    return {"dy": rows[:, 0].mean(), "err": rows[:, 1].mean(), "spread": spread,
            "d_dy": last[:, 0].mean() - first[:, 0].mean(),
            "d_err": last[:, 1].mean() - first[:, 1].mean()}


def soak(rallies, thoughts):
    """Play hard, then check that the four rules are still inside the bounds the
    whole calibration argument rests on.

    Every one of these is a *structural* claim, not a tuned number, which is why
    it is worth asserting after 2,000 frames rather than 6:

      * each retina row still delivers the sum |w| it was born with. That is what
        keeps the DSGC pool's mean input fixed, which is what keeps the brain off
        the 0.054 saturation edge. Renormalisation is a divide in float32 and the
        rule runs hundreds of times, so the question is whether the error walks.
      * theta only ever rose, and stayed under its ceiling. One-sidedness is what
        makes intrinsic plasticity incapable of making the brain louder.
      * both depression pools stayed <= 1. Bounded above by 1 is what makes them
        incapable of delivering more current than the calibrated weight - and it
        is the thing dishabituation could break, since it pushes d back up.
      * nothing saturated and nothing went silent, measured, not assumed.
    """
    fishsim.set_plasticity(ip=1, dep2=1, sens=1, hebb=1)
    r = build()
    print(f"  soak: {rallies} rallies x {thoughts} thoughts "
          f"= {rallies * thoughts} frames, all four rules on")
    rows = play(r, rallies, thoughts)
    s = r.sim
    h = s._hebb
    assert h is not None, "the Hebbian rule never armed - nothing was soaked"
    mag = np.abs(s.weights[:h["hi"]])
    l1 = np.add.reduceat(mag, h["starts"])
    drift = float(np.abs(l1 / h["l1"] - 1.0).max())
    detail = s.run(400)
    hz = detail["rates_hz"]
    hottest = max(hz.items(), key=lambda kv: kv[1])
    checks = [
        ("row L1 held", drift < 1e-3, f"worst row moved {drift:.2e} (gate 1e-3)"),
        ("theta one-sided", s.theta.min() >= 0.0 and s.theta.max() <= fishsim.IP_MAX,
         f"theta [{s.theta.min():.3f}, {s.theta.max():.3f}] mV, ceiling {fishsim.IP_MAX}"),
        ("d bounded", s.d.min() >= 0.0 and s.d.max() <= 1.0 + 1e-9,
         f"d [{s.d.min():.3f}, {s.d.max():.3f}]"),
        ("d2 bounded", s.d2.min() >= 0.0 and s.d2.max() <= 1.0 + 1e-9,
         f"d2 [{s.d2.min():.4f}, {s.d2.max():.4f}]"),
        ("weights finite", bool(np.isfinite(s.weights).all()), "no NaN, no inf"),
        ("not saturated", hottest[1] < 350.0, f"hottest {hottest[0]} {hottest[1]:.0f} Hz"),
        ("not silent", detail["spikes_per_sec"] > 0, f"{detail['spikes_per_sec']} spikes/s"),
        ("still plays", float(rows[:, 0].mean()) >= 0.08,
         f"mean |dy| {rows[:, 0].mean():.3f} (gate 0.08)"),
    ]
    print()
    for name, ok, note in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<18} {note}")
    print(f"    retina |w| spread {retina_spread(s):.2e}  ·  "
          f"{s._sens_n} startle restores")
    bad = [n for n, ok, _ in checks if not ok]
    print("\nsoak OK" if not bad else "\nsoak FAILED: " + ", ".join(bad))
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", type=int, default=6)
    ap.add_argument("--thoughts", type=int, default=20)
    ap.add_argument("--compare", action="store_true",
                    help="run it twice, with the Hebbian rule off and on")
    ap.add_argument("--soak", action="store_true",
                    help="play hard, then re-check the plasticity bounds")
    a = ap.parse_args()

    if a.soak:
        return soak(a.rallies, a.thoughts)

    if not a.compare:
        run_one("rally", a.rallies, a.thoughts, hebb=1)
        print("\nrally OK")
        return 0

    off = run_one("A", a.rallies, a.thoughts, hebb=0)
    print()
    on = run_one("B", a.rallies, a.thoughts, hebb=1)
    print(f"\n  hebbian off -> on:  |dy| {off['dy']:.4f} -> {on['dy']:.4f}"
          f"   paddle-ball {off['err']:.1f} -> {on['err']:.1f} px")
    print(f"  retina weights moved: {off['spread']:.2e} (off) vs {on['spread']:.2e} (on)")
    # The honest reading. A reward-free rule has no reason to improve pong, so
    # the question is only whether the difference is bigger than the noise the
    # same run shows against itself.
    d = abs(on["err"] - off["err"])
    noise = max(abs(off["d_err"]), abs(on["d_err"]))
    print(f"\n  difference {d:.1f} px vs within-run drift {noise:.1f} px -> "
          + ("bigger than the run's own drift; worth reporting"
             if d > noise else "inside the noise: no measurable effect on play"))
    print("  (the weights did change either way; what has not been shown is "
          "that the change helps, and nothing here rewards it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
