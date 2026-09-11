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
import json
from pathlib import Path

import numpy as np

REST_V = -52.0
THRESH_V = -45.0
TAU = 0.020
TAU_SYN = 0.002
TAU_REF = 0.002
DT = 0.0005
EPSP = 4.0  # mV delivered to a post target when a pre neuron spikes


def load_graph(graph_path, groups_path):
    npz = np.load(graph_path)
    groups_path = Path(groups_path)
    groups = json.loads(groups_path.read_text()) if groups_path.exists() else {}
    return {
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
        # synaptic input, low-passed
        self.v += self.isyn
        self.isyn *= np.exp(-dt / TAU_SYN)
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
        if fired.any():
            self.spike_counts[fired] += 1
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
            idx = self.groups.get(name)
            if idx is None or len(idx) == 0:
                return 0.0
            return int(window_spikes[idx].sum()) / (window_steps * dt)

        rates = {g: rate(g) for g in self.groups}
        return {
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
            "scatter": self.coords[firing].tolist()
            if len(firing) else [],
            "vetoed": 0,  # set by the roamer, not the brain
        }

    # -- behavior decode ----------------------------------------------------
    def decode(self, rates):
        """Map population rates onto the behaviors a larval fish actually has."""
        out = {"dx": 0.0, "dy": 0.0, "scroll": 0.0, "turn": 0.0, "escape": False}

        pairs = {
            "dsgc_up": (0.0, 1.0),
            "dsgc_down": (0.0, -1.0),
            "dsgc_left": (-1.0, 0.0),
            "dsgc_right": (1.0, 0.0),
        }
        best = None
        for name, (gx, gy) in pairs.items():
            r = rates.get(name, 0.0)
            if best is None or r > best[1]:
                best = ((gx, gy), r)
        (dx, dy), strongest = best
        scale = min(1.0, strongest / 40.0)  # 40 Hz drives full-speed steering
        out["dx"], out["dy"] = dx * scale, dy * scale

        # nMLF bout gate -> burst scrolling (a swim bout = a scroll burst)
        out["scroll"] = min(1.0, rates.get("nmlf", 0.0) / 30.0)

        # vSPN sets turn direction, signs against the DSGC winner
        out["turn"] = min(1.0, rates.get("vspn", 0.0) / 20.0) * (-1 if strongest else 1)

        # Mauthner: an escape is all-or-nothing and fast
        out["escape"] = rates.get("mauthner", 0.0) > 0.0
        return out


def smoke():
    """Drive the smoke graph with a rightward + a startle input and check."""
    graph = load_graph("build/graph.npz", "build/groups.json")
    sim = FishSim(graph)
    sim.set_drive(graph["groups"]["retina"], 120.0)
    sim.set_drive(graph["groups"]["dsgc_right"], 120.0)
    d1 = sim.run(400)
    dec1 = sim.decode(d1["rates_hz"])
    sim.set_drive(graph["groups"]["mauthner"], 0.0)  # not a real Mauthner drive
    assert sum(map(abs, (dec1["dx"], dec1["dy"]))) > 0, "rightward input should steer"
    print("smoke OK:", dec1)
    # startle: strong retina everywhere drives escape via the M-cell path
    sim2 = FishSim(graph, seed=3)
    sim2.set_drive(graph["groups"]["retina"], 400.0)
    d2 = sim2.run(400)
    dec2 = sim2.decode(d2["rates_hz"])
    print("startle OK:", dec2)


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
    print(json.dumps({k: v for k, v in d.items() if k != "scatter"}, indent=2))
    print("decode:", sim.decode(d["rates_hz"]))
    smoke()


if __name__ == "__main__":
    main()