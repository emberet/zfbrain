"""Find the weight scale that makes a graph behave like a brain.

A connectome is a wiring diagram, not a set of synaptic strengths: EM tells
you neuron A contacts neuron B with N synapses, not how many millivolts that
is worth. One global scale converts counts to mV, and the whole graph is
violently sensitive to it - at 208 synapses per neuron the same graph is silent
below ~0.027, and a 0.7% move near the working value takes the resting spinal
cord from 2 Hz to 20. This bisects that number until `fishsim.smoke()`'s
assertions hold, then writes it into graph.meta.json so every later run
(roam.py included) loads the calibrated brain.

    ZF_IP=1 ZF_DEP_U2=1 ZF_SENS=1 ZF_HEBB=1 python calibrate.py     # see below
    python calibrate.py --target 12        # aim for a different mean rate
    python calibrate.py --floor 8          # demand more margin under the motors
    python calibrate.py --dry              # print the scale, write nothing

Set the plasticity flags to whatever .env sets for the live fish. Every rule
can only ever deliver *less* than the calibrated weight, so a scale bisected
without them is systematically too weak for the fish that runs - and this is
not hypothetical: switching on the slow depression pool alone, a 3% cut in what
the eye delivers, took the spinal cord from 21.5 Hz to exactly 0.0 on a scale
that had just been certified alive.

The acceptance test is not "it runs": it is the same set fishsim asserts -
every readout population alive *with margin* rather than merely above zero,
none saturated, a still page producing no bout/turn/escape, a flash startling
the Mauthner cell. A scale that only satisfies the mean rate is reported as
such, not saved.
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
SETTLE_ROUNDS = 3   # how settle_depression() is called here and in smoke(), so
SETTLE_FRAMES = 4   # a scale that passes here still passes there. It pins the
                    # depression pools at their fixed point rather than running
                    # toward it, because running toward it takes ~370 frames -
                    # and the old code ran 12 and called that settled.
SETTLE = SETTLE_ROUNDS * SETTLE_FRAMES + 1      # frames one settle actually costs


class Probe:
    """Holds one FishSim over the graph (the CSR build is the expensive part -
    15 s at 39M edges) and rescales its weights in place per candidate."""

    def __init__(self, graph):
        self.graph = graph
        graph["meta"]["weight_scale"] = 1.0          # build at unit scale...
        t0 = time.time()
        self.sim = fs.FishSim(graph)
        self.base = self.sim.weights.copy()          # ...and keep the unscaled weights
        h = getattr(self.sim, "_hebb", None)
        self._hebb_l1 = h["l1"].copy() if h is not None else None
        self._hebb_w0 = float(h["w0"]) if h is not None else 0.0
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
        # d2 and theta were NOT reset here, and one Probe is reused for every
        # candidate, so with the plastic rules on each candidate inherited the
        # previous one's slow pool and threshold creep - the bisection would
        # have been scoring a monotonically more depressed brain as it walked,
        # which is exactly the sort of thing that makes a bisection converge on
        # a number that nothing can reproduce.
        sim.d2[:] = 1.0
        sim.theta[:] = 0.0
        # The Hebbian rule's whole safety argument is that it renormalises each
        # retina row back to the sum |w| it started with — and that sum is
        # cached at construction, which here happens at weight_scale = 1.0.
        # Rescaling the weights per candidate without rescaling the cache would
        # have the rule "restore" every row to ~20x the scale being tested, on
        # the first window, silently. The invariant is proportional to the
        # scale, so it rescales with it.
        h = getattr(sim, "_hebb", None)
        if h is not None:
            h["l1"] = self._hebb_l1 * scale
            h["w0"] = self._hebb_w0 * scale
            h["lo"] = np.float32(fs.HEBB_LO * h["w0"])
            h["hiw"] = np.float32(fs.HEBB_HI * h["w0"])
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
        # NOT `for _ in range(SETTLE): sim.run(400)`. That ran 2.4 s of
        # biological time against a depression pool whose time constant is 20 s,
        # so every candidate was scored on a brain with d still at ~0.96 - and
        # the scale this bisection returned was therefore the right scale for a
        # fish that had just woken up and the wrong one for the fish that runs.
        # settle_depression() pins the pools at their fixed point instead, for
        # about the same cost, so the number below is scored where the fish
        # lives. This is what silenced nMLF, vSPN and the spinal cord.
        d = sim.settle_depression(window_steps=400)
        out = {"scale": scale, "mean": d["spikes_per_sec"] / sim.n, "rates": d["rates_hz"],
               "decode": sim.decode(d["rates_hz"]),
               "d": float(sim.d[sim.plastic].mean()),
               "sec_per_frame": (time.time() - t0) / SETTLE}
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
              f"d {m['d']:.3f}  {m['sec_per_frame']:.2f}s/frame")
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
    ap.add_argument("--floor", type=float, default=5.0,
                    help="Hz every readout population must clear at rest, not merely "
                         "exceed zero (default 5)")
    ap.add_argument("--dry", action="store_true", help="print the result; write nothing")
    args = ap.parse_args()

    graph = fs.load_graph(args.graph, args.groups)
    meta_path = Path(args.graph).with_name("graph.meta.json")
    print(f"# {graph['meta'].get('label')}: {graph['meta']['neurons']:,} neurons, "
          f"{graph['meta']['synapses']:,} synapses, fan-in {graph['meta'].get('fanin')}")
    probe = Probe(graph)

    # Calibrate in the configuration the fish actually runs in. The plastic
    # rules default off in code and on in .env, and every one of them can only
    # ever deliver *less* than the calibrated weight (d <= 1, theta >= 0, the
    # Hebbian rule conserves its rows), so a scale bisected with them off is
    # systematically too weak for the fish that runs - which is the same shape
    # of mistake as settling for 2.4 s against a 20 s time constant, and it bit
    # just as hard: flipping ZF_DEP_U2 alone took the spinal cord to 0.0 Hz on
    # a scale that had just been certified alive. Calibrating with them on is
    # the safe direction of the two, because the off case is then merely
    # louder, and `hot`/`runaway` below still catch it if it is too loud.
    on = fs.plasticity_on()
    print(f"# plastic rules while calibrating: {', '.join(on) if on else 'none'} "
          f"(set ZF_IP/ZF_DEP_U2/ZF_SENS/ZF_HEBB to match the live fish)")
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
    # "> 0" is not a margin, and on this graph it is barely a fact. Measured on
    # the scale this bisection returned a week ago: the spinal cord fell from
    # 21.5 Hz to exactly 0.0 when the slow depression pool was switched on, and
    # to 12.8 when intrinsic plasticity moved the mean threshold by 0.011 mV.
    # The graph sits close enough to a bifurcation that the resting rate of a
    # readout population is not a property you can certify by watching it be
    # positive once. So the gate is a floor with room under it, and a scale that
    # only clears zero is reported as the near miss it is.
    #
    # The spinal cord alone is held to it. nMLF and vSPN idle at ~0.2 Hz on a
    # still page by design and answer in the 6-7 Hz range when the world moves -
    # the `moving` block above is where they are actually judged.
    thin = [g for g in ("spinal",) if 0.0 < rates.get(g, 0.0) < args.floor]
    ok = not dead and not thin and not hot and not runaway \
        and rates.get("mauthner", 0.0) < fs.ESCAPE_HZ \
        and flash >= fs.ESCAPE_HZ and not best["decode"]["escape"] and best["decode"]["scroll"] < 0.45
    if dead:
        print(f"# silent populations: {dead}")
    if thin:
        print(f"# alive but with no margin (< {args.floor:.0f} Hz at rest, and this graph "
              f"can lose that much to a 0.01 mV threshold change): "
              + ", ".join(f"{g} {rates.get(g, 0.0):.2f} Hz" for g in thin))
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
    # and into the per-graph store, which a later build of a *different* graph
    # (--smoke, say) cannot overwrite the way it overwrites the meta
    import build_graph as bg
    bg.write_calibration(meta_path.parent, meta)
    print(f"# wrote weight_scale to {meta_path} and {meta_path.parent / bg.CALIB}; "
          f"now run: python fishsim.py")


if __name__ == "__main__":
    main()
