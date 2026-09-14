#!/usr/bin/env python
"""Does the warm start land where waiting lands?

settle_depression() pins d to its analytic fixed point instead of running for
the minutes it would take to get there. That is only worth anything if the two
agree, so this measures both on the same brain and prints them side by side.
The long arm is the ground truth; the short one is what calibrate.py will use.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fishsim as fs           # noqa: E402

DRIVE_HZ, FLOW_HZ = 40.0, 12.0
POPS = ("retina", "dsgc_up", "dsgc_down", "nmlf", "vspn", "spinal", "other")


def drive(s):
    s.set_drive(s.groups.get("retina", []), DRIVE_HZ)
    s.set_drive([i for g, ix in s.groups.items() if g.startswith("dsgc")
                 for i in ix], FLOW_HZ)


def main():
    fs.set_plasticity(ip=False, dep2=False, sens=False, hebb=False)
    graph = fs.load_graph("build/graph.npz", "build/groups.json")

    # the long way: run until d has actually converged (5 tau_eff ~ 74 s)
    s = fs.FishSim(graph)
    t0 = time.time()
    for _ in range(400):
        drive(s)
        det_long = s.run(400)
    d_long = float(s.d[s.plastic].mean())
    t_long = time.time() - t0

    # the short way
    s2 = fs.FishSim(graph)
    t0 = time.time()
    drive(s2)
    det_warm = s2.settle_depression()
    d_warm = float(s2.d[s2.plastic].mean())
    t_warm = time.time() - t0

    print(f"\n  d after 400 frames (80 s biological)  {d_long:.4f}   "
          f"{t_long:5.0f} s wall")
    print(f"  d after settle_depression()           {d_warm:.4f}   "
          f"{t_warm:5.0f} s wall   ({t_long / max(t_warm, 1e-9):.0f}x faster)")
    print(f"  they differ by {abs(d_long - d_warm):.4f}\n")
    print(f"    {'population':<12} {'long run':>9} {'warm start':>11}")
    print("    " + "-" * 34)
    for g in POPS:
        print(f"    {g:<12} {det_long['rates_hz'].get(g, 0.0):9.2f} "
              f"{det_warm['rates_hz'].get(g, 0.0):11.2f}")
    ok = abs(d_long - d_warm) < 0.03
    print(f"\n  {'AGREE' if ok else 'DISAGREE'} on d (gate 0.03)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
