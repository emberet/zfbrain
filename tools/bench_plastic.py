#!/usr/bin/env python
"""What does each plasticity rule cost, on the real brain, per 400-step frame?

`fishsim.py --bench` builds a synthetic graph whose only group is "retina", so
the Hebbian rule would find no DSGC block, switch itself off, and report a cost
of zero. That is the wrong number. This loads build/graph.npz instead, where the
retina really does project 6.8M synapses onto the DSGC pool, and times one flag
at a time against the same seeded drive.

The budget the plan set: <=10% per frame without Hebbian, <=35% with it.

    .venv/bin/python tools/bench_plastic.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fishsim import FishSim, load_graph, set_plasticity  # noqa: E402

FRAMES = 6
STEPS = 400


def timed(graph, steps=STEPS, frames=FRAMES, seed=3):
    sim = FishSim(graph, seed=seed)
    rng = np.random.default_rng(seed)
    ret = graph["groups"]["retina"]
    # a drive that actually fires the retina, so the plastic paths are exercised
    # rather than skipped over a silent block
    sim.set_drive(ret, rng.uniform(5.0, 45.0, len(ret)))
    sim.run(40)  # warm the JIT and let the first Hebbian window allocate
    t0 = time.time()
    for _ in range(frames):
        sim.run(steps)
    return (time.time() - t0) / frames


def main():
    graph = load_graph("build/graph.npz", "build/groups.json")
    rows = []
    for tag, kw in [("off (today's brain)", dict(ip=0, dep2=0, sens=0, hebb=0)),
                    ("ZF_IP", dict(ip=1, dep2=0, sens=0, hebb=0)),
                    ("ZF_DEP_U2", dict(ip=0, dep2=1, sens=0, hebb=0)),
                    ("ZF_SENS", dict(ip=0, dep2=0, sens=1, hebb=0)),
                    ("ZF_HEBB", dict(ip=0, dep2=0, sens=0, hebb=1)),
                    ("all four", dict(ip=1, dep2=1, sens=1, hebb=1))]:
        set_plasticity(**kw)
        rows.append((tag, timed(graph)))
        t, base = rows[-1][1], rows[0][1]
        print(f"  {tag:<20} {t:6.3f} s/frame   {(t/base-1)*100:+6.1f}%")
    set_plasticity(ip=0, dep2=0, sens=0, hebb=0)
    print(f"\n({FRAMES} frames of {STEPS} steps each, real graph, same seeded drive)")


if __name__ == "__main__":
    sys.exit(main())
