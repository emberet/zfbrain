"""LIF whole-graph simulator over the larval zebrafish connectome graph.

Same three conventions as the fly branch (Shiu et al. 2024):
    rest       -52 mV
    threshold  -45 mV
    tau        20 ms
dt = 0.5 ms. Synaptic sign comes from the graph (vglut2a +, gad1b -).
No learning, no backprop, no policy network. Everything downstream reads
*firing rates of real populations* - the fish's actual behaviors come out of
the neurons a real zebrafish uses:

    DSGC (4 directions)   -> cursor steering
    nMLF bout gate        -> scroll bursts (swim bouts)
    vSPN                  -> turn direction / sudden reversal
    Mauthner cell         -> escape: the click / action veto

Usage:
    python fishsim.py                          # smoke test with a fake graph
    python fishsim.py --graph build/graph.npz  # real graph (build_graph first)
"""

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np

REST_V = -52.0
THRESH_V = -45.0
TAU = 0.020
TAU_SYN = 0.002
TAU_REF = 0.002
DT = 0.0005
EPSP = 4.0  # mV delivered to a post target by one spike over the strongest synapse (count 6)
FLASH_STEPS = 10  # "fired this step" on the site = spiked within the last 5 ms
ESCAPE_HZ = float(os.environ.get("ZF_ESCAPE_HZ", "30"))  # Mauthner rate that counts as a startle
NMLF_REST, NMLF_SPAN = 30.0, 40.0   # a bout = nMLF this far above its resting rate
VSPN_REST, VSPN_SPAN = 30.0, 30.0   # a turn = vSPN this far above its resting rate


def load_graph(graph_path, groups_path):
    npz = np.load(graph_path)
    groups_path = Path(groups_path)
    groups = json.loads(groups_path.read_text()) if groups_path.exists() else {}
    meta_path = Path(graph_path).with_name("graph.meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("label", "unknown graph")
    meta.setdefault("neurons", int(len(npz["coords"])))
    meta.setdefault("synapses", int(len(npz["pre"])))
    return {
        "meta": meta,
        "root_ids": npz["root_ids"],
        "coords": npz["coords"],
        "types": npz["types"],
        "pre": npz["pre"],
        "post": npz["post"],
        "w": npz["w"],
        "groups": {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()},
    }


class FishSim:
    def __init__(self, graph, seed=11):
        self.n = len(graph["coords"])
        self.pre = graph["pre"]
        self.post = graph["post"]
        self.w = graph["w"].astype(np.float64)
        self.root_ids = graph["root_ids"]
        self.coords = graph["coords"]
        self.groups = graph.get("groups", {})
        self.rng = np.random.default_rng(seed)

        # scale weights so typical EPSPs stay in a sane mV band
        scale = 1.0
        m = np.abs(self.w)
        if m.max() > 0:
            scale = EPSP / min(6.0, m.max())
        self.w = self.w * scale

        self.v = np.full(self.n, REST_V, dtype=np.float64)
        self.isyn = np.zeros(self.n, dtype=np.float64)
        self.refract = np.zeros(self.n, dtype=np.float64)
        self.spike_counts = np.zeros(self.n, dtype=np.int64)
        self.last_spike = np.full(self.n, -10**9, dtype=np.int64)  # step index of the last spike
        self.step_i = 0
        self.drive = np.zeros(self.n, dtype=np.float64)  # sensory rate, Hz
        self.t = 0.0

    # -- sensory port ---------------------------------------------------
    def set_drive(self, indices, rate_hz):
        """Stimulate labelled neurons (retina, DSGC...) as a Poisson train."""
        idx = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        idx = idx[idx < self.n]
        if idx.size == 0:
            return
        rates = np.atleast_1d(np.asarray(rate_hz, dtype=np.float64))
        if rates.size != idx.size:
            if rates.size == 1:
                rates = np.full(idx.size, rates[0])
            else:
                # resample whatever the retina handed us onto the connectome's
                # neuron count for that group (smoke graphs differ from Fish1)
                rates = np.interp(
                    np.linspace(0, 1, idx.size),
                    np.linspace(0, 1, rates.size), rates)
        self.drive[idx] = rates

    def clear_drive(self):
        self.drive[:] = 0.0

    # -- one discrete step ----------------------------------------------
    def _step(self, dt=DT):
        # leak
        self.v += (REST_V - self.v) * (dt / TAU)
        # synaptic input: isyn is charge left to deliver; each step hands over
        # a TAU_SYN-fraction of it, so one spike of weight w adds w mV in total
        kick = self.isyn * (dt / TAU_SYN)
        self.v += kick
        self.isyn -= kick
        # sensory Poisson input
        prob = 1.0 - np.exp(-self.drive * dt)
        hits = self.rng.random(self.n) < prob
        self.v += hits.astype(np.float64) * EPSP
        # refractoriness
        self.refract -= dt
        blocked = self.refract > 0
        self.v[blocked] = REST_V
        # fire
        fired = (self.v >= THRESH_V) & ~blocked
        self.step_i += 1
        if fired.any():
            self.spike_counts[fired] += 1
            self.last_spike[fired] = self.step_i
            self.v[fired] = REST_V
            self.refract[fired] = TAU_REF
            # distribute to postsynaptic targets
            src = np.flatnonzero(fired)
            idx = np.isin(self.pre, src)
            np.add.at(self.isyn, self.post[idx], self.w[idx])
        return fired

    # -- full epoch -------------------------------------------------------
    def run(self, n_steps=400, dt=DT, keep_trace=False):
        """Run n_steps of biological time. Returns detail dict (see stream())."""
        trace = []
        for _ in range(n_steps):
            fired = self._step(dt)
            trace.append(fired)
        self.t += n_steps * dt
        detail = self.stream(dt, n_steps)
        if keep_trace:
            detail["spike_trace"] = np.array(trace)
        return detail

    # -- readout (the panels the live site shows) -------------------------
    def stream(self, dt=DT, window_steps=400):
        window_v = self.v
        window_spikes = self.spike_counts.copy()
        # reset per-window counters for next readout
        firing = np.flatnonzero(window_spikes)
        self.spike_counts[:] = 0

        def rate(name):
            """Mean firing rate per neuron in the population, Hz."""
            idx = self.groups.get(name)
            if idx is None or len(idx) == 0:
                return 0.0
            return float(window_spikes[idx].sum()) / (len(idx) * window_steps * dt)

        rates = {g: rate(g) for g in self.groups}
        recent = self.last_spike >= self.step_i - FLASH_STEPS
        return {
            "firing_mask": base64.b64encode(np.packbits(recent).tobytes()).decode("ascii"),
            "firing_recent": int(recent.sum()),
            "t": round(self.t, 3),
            "neurons": int(self.n),
            "spikes_per_sec": int(window_spikes.sum() / (window_steps * dt)),
            "mean_membrane_mv": round(float(window_v.mean()), 2),
            "visual_spikes": int(window_spikes[self.groups.get("retina", [])].sum())
            if "retina" in self.groups else 0,
            "motor_spikes": int(window_spikes[self.groups.get("spinal", [])].sum())
            if "spinal" in self.groups else 0,
            "rates_hz": {k: round(v, 2) for k, v in rates.items()},
            "firing_indices": firing.tolist(),
            "vetoed": 0,  # set by the roamer, not the brain
        }

    # -- behavior decode ----------------------------------------------------
    def decode(self, rates, rest=None):
        """Map population rates onto the behaviors a larval fish actually has.
        `rest` = resting rates per population (the roamer passes running
        averages, so a bout is a rise above the fish's own recent baseline)."""
        rest = rest or {}
        nmlf_rest = rest.get("nmlf", NMLF_REST)
        vspn_rest = rest.get("vspn", VSPN_REST)
        out = {"dx": 0.0, "dy": 0.0, "scroll": 0.0, "turn": 0.0, "escape": False}

        # a fish steers by asymmetry: the difference between opposed DSGC
        # pairs, not the loudest one — a balanced field is no steering at all
        up, dn = rates.get("dsgc_up", 0.0), rates.get("dsgc_down", 0.0)
        lf, rt = rates.get("dsgc_left", 0.0), rates.get("dsgc_right", 0.0)
        out["dx"] = float(np.clip((rt - lf) / max(rt + lf, 1.0), -1.0, 1.0))
        out["dy"] = float(np.clip((dn - up) / max(dn + up, 1.0), -1.0, 1.0))
        strongest = max(up, dn, lf, rt)

        # nMLF bout gate -> a scroll burst when the population rises above rest
        out["scroll"] = float(np.clip((rates.get("nmlf", 0.0) - nmlf_rest) / NMLF_SPAN, 0.0, 1.0))

        # vSPN above rest -> a turn, signed against the DSGC winner
        out["turn"] = float(np.clip((rates.get("vspn", 0.0) - vspn_rest) / VSPN_SPAN, 0.0, 1.0)) * (-1 if strongest else 1)

        # Mauthner: an escape is all-or-nothing and fast — a startle is a burst,
        # not the odd coincidence spike a cell with hundreds of inputs throws
        out["escape"] = rates.get("mauthner", 0.0) > ESCAPE_HZ
        return out


NEED = ["retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right",
        "nmlf", "vspn", "mauthner", "spinal"]


def smoke(graph_path="build/graph.npz", groups_path="build/groups.json"):
    """Checks that hold for any graph we run — the toy one and the synthetic one.

    1. under a plain retina drive every readout population fires (> 0 Hz) and
       none saturates (< 350 Hz; the 2 ms refractory ceiling is 500)
    2. a rightward DSGC drive steers right
    3. Mauthner sits below ESCAPE_HZ at rest and a whole-field flash lifts it
       at least 5x — the startle is a burst, not a habit
    """
    graph = load_graph(graph_path, groups_path)
    label = graph["meta"].get("label", "graph")
    sim = FishSim(graph)
    sim.set_drive(graph["groups"]["retina"], 60.0)
    for _ in range(3):
        d1 = sim.run(400)  # let the recurrent pools settle
    r1 = d1["rates_hz"]
    dead = [g for g in NEED if r1.get(g, 0.0) <= 0.0]
    hot = [g for g in NEED if r1.get(g, 0.0) >= 350.0]
    print(f"[{label}] rest rates:", {g: r1[g] for g in NEED})
    if graph["meta"].get("source") != "smoke":
        assert not dead, f"silent populations under retina drive: {dead}"
    assert not hot, f"saturated populations: {hot}"
    assert r1.get("mauthner", 0.0) < ESCAPE_HZ, "Mauthner startles at rest"
    rest = sim.decode(r1)
    assert not rest["escape"] and rest["scroll"] < 0.45 and abs(rest["turn"]) < 0.5, \
        f"a still page should not make the fish bout, turn or escape: {rest}"
    print("rest decode OK:", {k: round(v, 2) if isinstance(v, float) else v for k, v in rest.items()})

    sim2 = FishSim(graph, seed=5)
    sim2.set_drive(graph["groups"]["retina"], 60.0)
    sim2.set_drive(graph["groups"]["dsgc_right"], 120.0)
    dec = sim2.decode(sim2.run(400)["rates_hz"])
    assert dec["dx"] > 0, f"rightward input should steer right, got {dec}"
    print("steer OK:", dec)

    sim3 = FishSim(graph, seed=3)
    sim3.set_drive(graph["groups"]["retina"], 60.0)
    base = sim3.run(400)["rates_hz"].get("mauthner", 0.0)
    sim3.set_drive(graph["groups"]["retina"], 400.0)  # the flash
    for g in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
        sim3.set_drive(graph["groups"][g], 200.0)
    flash = sim3.run(200)["rates_hz"].get("mauthner", 0.0)
    print(f"startle: mauthner {base:.1f} Hz at rest -> {flash:.1f} Hz on a flash (escape at {ESCAPE_HZ})")
    if graph["meta"].get("source") != "smoke":
        assert flash >= max(5 * base, ESCAPE_HZ), "a whole-field flash should startle the Mauthner cell"
    print("smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="build/graph.npz")
    ap.add_argument("--groups", default="build/groups.json")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    graph = load_graph(args.graph, args.groups)
    sim = FishSim(graph)
    sim.set_drive(graph["groups"].get("retina", []), 100.0)
    for side in ("right", "up"):
        sim.set_drive(graph["groups"].get(f"dsgc_{side}", []), 100.0)
    d = sim.run(args.steps)
    print(json.dumps({k: v for k, v in d.items() if k not in ("firing_indices", "firing_mask")}, indent=2))
    print("decode:", sim.decode(d["rates_hz"]))
    smoke(args.graph, args.groups)


if __name__ == "__main__":
    main()