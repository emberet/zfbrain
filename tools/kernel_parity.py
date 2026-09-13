#!/usr/bin/env python
"""Do the numba kernels and the numpy fallback still compute the same brain?

fishsim has two implementations of every rule: the @njit kernels, and the
vectorised numpy path that runs where numba is missing. Only one of them is
ever exercised on this machine, so the fallback is the one most likely to drift
silently — and it drifts on the machines we cannot see.

The two paths cannot be compared under normal drive, because they draw their
Poisson sensory hits from different generators (np.random inside the kernel,
self.rng outside). So this pins the stochastic term instead: hit_p is set to
exactly 0.0 or 1.0, and `random() < 1.0` is true in both generators. Both paths
then become deterministic and must agree exactly, not statistically.

    python tools/kernel_parity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fishsim  # noqa: E402
from fishsim import FishSim, load_graph, set_plasticity  # noqa: E402


def toy(n=4000, deg=40, seed=5):
    """A small graph with the group names the rules key off, so the retina
    block is contiguous exactly as it is in the real one."""
    rng = np.random.default_rng(seed)
    e = n * deg
    n_ret, n_ds = n // 8, n // 8
    return {
        "root_ids": np.arange(n),
        "coords": rng.normal(0, 1, (n, 3)).astype(np.float32),
        "types": np.array(["unknown"]),
        "pre": np.repeat(np.arange(n, dtype=np.int64), deg),
        # the retina's edges must land on the DSGC block, as build_graph makes them
        "post": np.concatenate([
            rng.integers(n_ret, n_ret + n_ds, n_ret * deg, dtype=np.int64),
            rng.integers(0, n, e - n_ret * deg, dtype=np.int64)]),
        "w": (rng.choice([-1.0, 1.0], e) * 6.0).astype(np.float32),
        "groups": {"retina": np.arange(0, n_ret),
                   "dsgc_up": np.arange(n_ret, n_ret + n_ds // 2),
                   "dsgc_down": np.arange(n_ret + n_ds // 2, n_ret + n_ds),
                   "mauthner": np.arange(n - 40, n)},
        "meta": {"source": "parity", "weight_scale": 0.15},
    }


def run(graph, use_numba, steps=240, seed=11):
    was, fishsim.HAVE_NUMBA = fishsim.HAVE_NUMBA, use_numba
    try:
        sim = FishSim(graph, seed=seed)
        # pin the stochastic term: hit_p of exactly 1.0 fires every step in both
        # generators, 0.0 in neither. No other source of randomness exists.
        sim.hit_p[:] = 0.0
        sim.hit_p[: len(graph["groups"]["retina"])] = 1.0
        sim.hit_p[np.arange(0, sim.n, 7)] = 1.0
        for _ in range(steps // 40):
            sim.run(40)
        return sim
    finally:
        fishsim.HAVE_NUMBA = was


def compare(tag, a, b):
    fields = [("v", a.v, b.v), ("isyn", a.isyn, b.isyn), ("d", a.d, b.d),
              ("d2", a.d2, b.d2), ("theta", a.theta, b.theta),
              ("last_spike", a.last_spike.astype(float), b.last_spike.astype(float)),
              ("weights", a.weights.astype(np.float64), b.weights.astype(np.float64))]
    ok = True
    print(f"  {tag}")
    for name, x, y in fields:
        scale = max(float(np.abs(x).max()), 1e-12)
        err = float(np.abs(x - y).max()) / scale
        good = err < 1e-9
        ok &= good
        print(f"    {'ok  ' if good else 'DIFF'} {name:<11} max rel diff {err:.2e}")
    return ok


def main():
    graph = toy()
    all_ok = True
    for tag, kw in [("all rules off (today's brain)", dict(ip=0, dep2=0, sens=0, hebb=0)),
                    ("intrinsic", dict(ip=1, dep2=0, sens=0, hebb=0)),
                    ("slow depression pool", dict(ip=0, dep2=1, sens=0, hebb=0)),
                    ("dishabituation", dict(ip=0, dep2=0, sens=1, hebb=0)),
                    ("hebbian", dict(ip=0, dep2=0, sens=0, hebb=1)),
                    ("all four", dict(ip=1, dep2=1, sens=1, hebb=1))]:
        set_plasticity(**kw)
        a, b = run(graph, True), run(graph, False)
        all_ok &= compare(tag, a, b)
    print()
    print("kernel parity OK" if all_ok else "KERNEL PARITY FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
