#!/usr/bin/env python
"""Does the brain go quiet if you leave it running?

The soak asks "is the graph silent?" by summing spikes over every neuron. That
number is dominated by the retina, which fires because `set_drive` injects
current into it directly - it does not care what any weight is worth. So the
soak's silence gate cannot fail for the one reason that matters: the brain
*downstream* of the driven populations going dark while the driven ones carry
on. This measures that separately.

    python tools/silence.py               # all four rules, ~20 min of fish
    python tools/silence.py --minutes 5
    python tools/silence.py --plain       # no plasticity, for the comparison
    python tools/silence.py --sweep       # where does the delivered factor kill it?
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fishsim as fs           # noqa: E402

DRIVE_HZ = 40.0                # a page's worth of retina drive
FLOW_HZ = 12.0                 # and a little optic flow, as a rally gives

# the populations that only ever fire because something upstream made them.
# If these are at zero while the retina is at 11 Hz, the graph is not quiet -
# it is disconnected.
#
# mauthner is deliberately NOT here. fishsim.py:769 says "the Mauthner cell is
# meant to be silent between escapes", and the sweep confirms it: 0.00 Hz at
# every delivered factor including 1.0, on the fully calibrated brain. A gate
# that called that a failure would be red on a healthy fish.
DOWNSTREAM = ("nmlf", "vspn", "spinal", "other")


def sweep():
    """At what delivered factor does the brain behind the eyes go out?

    calibrate.py's "silent at 0.027" is a *global* rescale of every weight in the
    graph. `d * d2` is not that: it scales only what leaves the sensory
    populations, and leaves the rest of the brain's internal wiring alone. So the
    0.027 figure is the right order of magnitude and the wrong number, and the
    only way to know the real floor is to multiply that block and look.

    Scaling the weights is exactly what the pools do - `_propagate` hoists
    `d[s] * d2[s]` and multiplies the whole row by it - so this measures the
    thing itself, not a model of it.

    The fast pool has to be pinned for the x-axis to mean anything. DEP_TAU is
    20 s and a settle is a couple of biological seconds, so `d` would still be
    drifting down from 1.0 at every row and the real factor would be `f` times
    an unknown. Set DEP_U to zero and `d` stays at 1.0, so the delivered factor
    is exactly `f`. The kernels read the constant at call time, so assigning the
    module global is enough.
    """
    fs.set_plasticity(ip=False, dep2=False, sens=False, hebb=False)
    fs.DEP_U = 0.0
    graph = fs.load_graph("build/graph.npz", "build/groups.json")
    s = fs.FishSim(graph)
    retina = s.groups.get("retina", [])
    dsgc = [i for g, ix in s.groups.items() if g.startswith("dsgc") for i in ix]

    # the plastic neurons' outgoing CSR rows. They are the contiguous block
    # [0, n_plastic) - asserted, because the whole trick depends on it.
    plastic_ix = np.flatnonzero(s.plastic)
    n_plastic = len(plastic_ix)
    assert (plastic_ix == np.arange(n_plastic)).all(), "plastic set is not a prefix"
    end = int(s.indptr[n_plastic])
    w0 = s.weights[:end].copy()
    print(f"  plastic block: {n_plastic} neurons, {end} outgoing synapses "
          f"({end / len(s.weights):.1%} of the graph)\n")

    print(f"    {'d*d2':>6}  {'retina':>7} {'dsgc':>6}  "
          f"{'nmlf':>6} {'vspn':>6} {'mauth':>6} {'spinal':>7} {'other':>7}")
    print("    " + "-" * 64)
    floor = None
    for f in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4):
        s.weights[:end] = w0 * f
        s.v[:] = fs.REST_V                      # a clean start each time, so the
        s.isyn[:] = 0.0                         # previous factor cannot leak in
        s.d[:] = 1.0
        for _ in range(12):                     # settle, then measure the last one
            s.set_drive(retina, DRIVE_HZ)
            s.set_drive(dsgc, FLOW_HZ)
            det = s.run(400)
        r = det["rates_hz"]
        down = [r.get(g, 0.0) for g in DOWNSTREAM]
        if floor is None and min(down) <= 0.001:
            floor = f
        print(f"    {f:6.2f}  {r.get('retina', 0):7.2f} "
              f"{(r.get('dsgc_up', 0) + r.get('dsgc_down', 0)) / 2:6.2f}  "
              f"{r.get('nmlf', 0):6.2f} {r.get('vspn', 0):6.2f} "
              f"{r.get('mauthner', 0):6.2f} {r.get('spinal', 0):7.2f} "
              f"{r.get('other', 0):7.2f}")
    s.weights[:end] = w0
    print(f"\n  first factor with a downstream population at zero: "
          f"{'none in range' if floor is None else f'{floor:.2f}'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0,
                    help="biological minutes to run (DEP_TAU2 is 10 of them)")
    ap.add_argument("--plain", action="store_true", help="all rules off")
    ap.add_argument("--sweep", action="store_true",
                    help="scale the sensory block instead, and find the floor")
    a = ap.parse_args()

    if a.sweep:
        return sweep()

    fs.set_plasticity(ip=not a.plain, dep2=not a.plain,
                      sens=not a.plain, hebb=not a.plain)
    print(f"  rules: {fs.plasticity_on() or ['none']}")
    graph = fs.load_graph("build/graph.npz", "build/groups.json")
    s = fs.FishSim(graph)
    retina = s.groups.get("retina", [])
    dsgc = [i for g, ix in s.groups.items() if g.startswith("dsgc") for i in ix]

    n_steps = 400                       # one thought, same as roam.py
    frames = int(a.minutes * 60 / (n_steps * 0.0005))
    print(f"  {frames} frames = {a.minutes:g} biological minutes\n")
    print(f"    {'min':>5}  {'retina':>7} {'dsgc':>6}  "
          f"{'nmlf':>6} {'vspn':>6} {'spinal':>7} {'other':>7}  "
          f"{'d':>5} {'d2':>5} {'d*d2':>5}")
    print("    " + "-" * 76)

    t0 = time.time()
    for f in range(frames):
        s.set_drive(retina, DRIVE_HZ)
        s.set_drive(dsgc, FLOW_HZ)
        det = s.run(n_steps)
        if f % max(1, frames // 20) and f != frames - 1:
            continue
        r = det["rates_hz"]
        d = float(s.d[s.plastic].mean())
        d2 = float(s.d2[s.plastic].mean())
        print(f"    {f * n_steps * 0.0005 / 60:5.1f}  "
              f"{r.get('retina', 0):7.2f} "
              f"{(r.get('dsgc_up', 0) + r.get('dsgc_down', 0)) / 2:6.2f}  "
              f"{r.get('nmlf', 0):6.2f} {r.get('vspn', 0):6.2f} "
              f"{r.get('spinal', 0):7.2f} {r.get('other', 0):7.2f}  "
              f"{d:5.3f} {d2:5.3f} {d * d2:5.3f}")

    r = det["rates_hz"]
    dead = [g for g in DOWNSTREAM if r.get(g, 0.0) <= 0.001]
    print(f"\n  ran in {time.time() - t0:.0f} s wall clock")
    print(f"  downstream populations at exactly zero: "
          f"{', '.join(dead) if dead else 'none'}")
    scale = s.meta.get("weight_scale", 0.050141)
    print(f"  delivered factor {s.d[s.plastic].mean() * s.d2[s.plastic].mean():.4f}"
          f"  ->  effective scale {s.d[s.plastic].mean() * s.d2[s.plastic].mean() * scale:.5f}"
          f"  (calibrate.py: silent below 0.027)")
    return 1 if dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
