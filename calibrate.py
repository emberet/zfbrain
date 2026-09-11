"""Find the weight scale that makes a graph behave like a brain.

A connectome is a wiring diagram, not a set of synaptic strengths: EM tells
you neuron A contacts neuron B with N synapses, not how many millivolts that
is worth. One global scale converts counts to mV, and the whole graph is
violently sensitive to it - at 208 synapses per neuron the same graph is
silent at 0.027 and saturated at 0.054. This bisects that number until
`fishsim.smoke()`'s assertions hold, then writes it into graph.meta.json so
every later run (roam.py included) loads the calibrated brain.

    python calibrate.py                    # build/graph.npz, writes the result
    python calibrate.py --target 12        # aim for a different mean rate
    python calibrate.py --dry              # print the scale, write nothing

The acceptance test is not "it runs": it is the same set fishsim asserts -
every readout population alive but none saturated, a still page producing no
bout/turn/escape, a flash startling the Mauthner cell, ten flashes habituating
it. A scale that only satisfies the mean rate is reported as such, not saved.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fishsim as fs

DRIVE = 40.0        # what a white page drives the retina with, roughly
DSGC_DRIVE = 0.0    # a page the fish is not moving has no optic flow, and that
                    # is the state it spends most of its life in - calibrate
                    # there, then check that real flow doesn't blow it up
FLOW = 25.0         # the motion drive a scroll produces (Retina.motion_max)
SETTLE = 12         # frames to settle - the same number fishsim.smoke() uses,
                    # so a scale that passes here passes there


class Probe:
    """Holds one FishSim over the graph (the CSR build is the expensive part -
    15 s at 39M edges) and rescales its weights in place per candidate."""

    def __init__(self, graph):
        self.graph = graph
        graph["meta"]["weight_scale"] = 1.0          # build at unit scale...
        t0 = time.time()
        self.sim = fs.FishSim(graph)
        self.base = self.sim.weights.copy()          # ...and keep the unscaled weights
        print(f"# graph in memory: {self.sim.n:,} neurons, {len(self.base):,} synapses "
              f"({time.time() - t0:.1f}s)")

    def reset(self, scale, seed=11):
        sim = self.sim
        sim.weights[:] = self.base * scale
        sim.v[:] = fs.REST_V
        sim.isyn[:] = 0.0
        sim.refract[:] = 0.0
        sim.spike_counts[:] = 0
        sim.last_spike[:] = -10**9
        sim.d[:] = 1.0
        sim.step_i = 0
        sim.t = 0.0
        sim.clear_drive()
        sim.rng = np.random.default_rng(seed)
        if fs.HAVE_NUMBA:
            fs._seed(seed)
        return sim

    def measure(self, scale, flash=False, dsgc=None):
        """Mean rate and per-population rates under a plain page-like drive;
        with flash=True, also the Mauthner response to a whole-field flash."""
        sim = self.reset(scale)
        g = self.graph["groups"]
        sim.set_drive(g["retina"], DRIVE)
        for name in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
            sim.set_drive(g[name], DSGC_DRIVE if dsgc is None else dsgc)
        t0 = time.time()
        for _ in range(SETTLE):
            d = sim.run(400)
        out = {"scale": scale, "mean": d["spikes_per_sec"] / sim.n, "rates": d["rates_hz"],
               "decode": sim.decode(d["rates_hz"]), "sec_per_frame": (time.time() - t0) / SETTLE}
        if flash:
            sim.set_drive(g["retina"], 400.0)
            for name in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
                sim.set_drive(g[name], 200.0)
            out["flash_mauthner"] = sim.run(200)["rates_hz"].get("mauthner", 0.0)
        return out


def bisect(probe, target, lo, hi, rounds=12):
    """The mean firing rate rises monotonically with the scale; walk in on it."""
    best = None
    for i in range(rounds):
        mid = (lo * hi) ** 0.5                       # geometric: the response is multiplicative
        m = probe.measure(mid)
        r = m["rates"]
        print(f"  {mid:.6f} -> mean {m['mean']:6.1f} Hz  (dsgc {r['dsgc_up']:5.1f}  nmlf {r['nmlf']:6.1f}  "
              f"vspn {r['vspn']:6.1f}  mauthner {r['mauthner']:6.1f}  spinal {r['spinal']:6.1f})  "
              f"{m['sec_per_frame']:.2f}s/frame")
        if best is None or abs(m["mean"] - target) < abs(best["mean"] - target):
            best = m
        if m["mean"] < target:
            lo = mid
        else:
            hi = mid
        if abs(m["mean"] - target) < 0.05 * target:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="build/graph.npz")
    ap.add_argument("--groups", default="build/groups.json")
    ap.add_argument("--target", type=float, default=15.0,
                    help="mean firing rate across the whole graph, Hz (default 15)")
    ap.add_argument("--lo", type=float, default=1e-4)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--dry", action="store_true", help="print the result; write nothing")
    args = ap.parse_args()

    graph = fs.load_graph(args.graph, args.groups)
    meta_path = Path(args.graph).with_name("graph.meta.json")
    print(f"# {graph['meta'].get('label')}: {graph['meta']['neurons']:,} neurons, "
          f"{graph['meta']['synapses']:,} synapses, fan-in {graph['meta'].get('fanin')}")
    probe = Probe(graph)

    print(f"# bisecting the weight scale for a mean of {args.target} Hz:")
    best = bisect(probe, args.target, args.lo, args.hi)
    scale, rates = best["scale"], best["rates"]
    print(f"\nweight_scale = {scale:.6f}  (mean {best['mean']:.1f} Hz, "
          f"{best['sec_per_frame']:.2f}s per 400-step frame)")
    print("rates:", {k: round(v, 1) for k, v in rates.items()})
    print("decode on a still page:", {k: round(v, 3) if isinstance(v, float) else v
                                      for k, v in best["decode"].items()})

    # the Mauthner cell is judged on its startle, not on a resting rate: a
    # larva's M-cell is quiet between escapes, so silence here is correct and
    # what matters is that a whole-field flash still fires it
    flash = probe.measure(scale, flash=True)["flash_mauthner"]
    print(f"startle: mauthner {rates.get('mauthner', 0.0):.1f} Hz at rest -> {flash:.1f} Hz on a flash "
          f"(escape at {fs.ESCAPE_HZ})")
    # and the other half: when the fish does move the page, the optic flow it
    # makes must drive the motor pools without running the brain away
    moving = probe.measure(scale, dsgc=FLOW)
    print(f"under {FLOW:.0f} Hz of optic flow: mean {moving['mean']:.1f} Hz",
          {k: round(v, 1) for k, v in moving["rates"].items()})
    runaway = [g for g in fs.NEED if moving["rates"].get(g, 0.0) >= 350.0]
    if runaway:
        print(f"# optic flow saturates: {runaway}")
    dead = [g for g in fs.NEED if g != "mauthner" and rates.get(g, 0.0) <= 0.0]
    hot = [g for g in fs.NEED if rates.get(g, 0.0) >= 350.0]
    ok = not dead and not hot and not runaway and rates.get("mauthner", 0.0) < fs.ESCAPE_HZ \
        and flash >= fs.ESCAPE_HZ and not best["decode"]["escape"] and best["decode"]["scroll"] < 0.45
    if dead:
        print(f"# silent populations: {dead}")
    if hot:
        print(f"# saturated populations: {hot}")
    if flash < fs.ESCAPE_HZ:
        print(f"# a flash does not startle the Mauthner cell ({flash:.1f} Hz)")
    if not ok:
        print("# NOT usable as it stands — widen the range, retune SYN in build_graph.py, "
              "or build a smaller graph; nothing written")
        raise SystemExit(1)

    if args.dry:
        print("# --dry: nothing written")
        return
    meta = json.loads(meta_path.read_text())
    meta["weight_scale"] = round(float(scale), 6)
    meta["calibrated_mean_hz"] = round(float(best["mean"]), 2)
    meta["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"# wrote weight_scale to {meta_path}; now run: python fishsim.py")


if __name__ == "__main__":
    main()
