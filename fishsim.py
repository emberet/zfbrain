"""LIF whole-graph simulator over the larval zebrafish connectome graph.

Same three conventions as the fly branch (Shiu et al. 2024):
    rest       -52 mV
    threshold  -45 mV
    tau        20 ms
dt = 0.5 ms. Synaptic sign comes from the graph (vglut2a +, gad1b -).
No backprop, no policy network. Everything downstream reads *firing rates of
real populations* - the fish's behaviours come out of the neurons a real
zebrafish uses:

    DSGC / pretectal DS (4 directions) -> cursor steering (by asymmetry)
    nMLF                               -> swim bouts (scroll bursts)
    vSPN                               -> turns
    Mauthner cell                      -> escape (a burst, not a spike)

The only plasticity is the real one a larva has on its sensory inputs:
short-term synaptic depression (habituation) - every spike of a sensory
neuron uses up a fraction U of its synaptic resource, which recovers with
TAU_D. Repeated flashes stop startling the Mauthner cell; a page stared at
for a minute drives the brain less. No reward signal is invented.

Scale: the graph is CSR by presynaptic neuron and propagation is
event-driven (only the neurons that fired touch their outgoing synapses), in
numba when available - measured ~0.2 s per 400-step frame at 180k neurons /
30M synapses on an M4, ~1.5 s in the numpy fallback.

Usage:
    python fishsim.py                          # checks on build/graph.npz
    python fishsim.py --bench 180000           # in-memory whole-brain timing
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # the numpy path below is the same maths, ~7x slower
    HAVE_NUMBA = False

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (a and callable(a[0])) else a[0]

REST_V = -52.0
THRESH_V = -45.0
TAU = 0.020
TAU_SYN = 0.002
TAU_REF = 0.002
DT = 0.0005
EPSP = 4.0  # mV delivered to a post target by one spike over the strongest synapse (count 6)
FLASH_STEPS = 10  # "fired this step" on the site = spiked within the last 5 ms
ESCAPE_HZ = float(os.environ.get("ZF_ESCAPE_HZ", "30"))  # Mauthner rate that counts as a startle
# An escape is a burst: Mauthner must rise above BOTH its own running baseline
# (the fish's own recent average) AND the absolute floor.  A still page keeps
# Mauthner steady below ESCAPE_HZ; a flash lifts it ≥5×, well above the burst
# multiplier, so escape fires.  A hot page that keeps Mauthner at 40 Hz
# continuously will NOT trigger escapes, because the baseline climbs with it.
MAUTHNER_REST = float(os.environ.get("ZF_MAUTHNER_REST", "8.0"))  # fallback before baseline exists
ESCAPE_BURST = float(os.environ.get("ZF_ESCAPE_BURST", "1.8"))    # × baseline = a true startle
# A bout is nMLF rising above its own resting rate. How far above cannot be a
# fixed number of Hz: the resting rate depends on the graph (3 Hz here, 30 Hz
# on the small one), so the span scales with it, with a floor so a nearly
# silent population still needs a real excursion.
NMLF_REST, VSPN_REST = 30.0, 30.0   # fallbacks when the roamer has no baseline yet
SPAN_FACTOR, MIN_SPAN = 1.5, 8.0
# habituation: short-term depression on sensory-input synapses
DEP_U = float(os.environ.get("ZF_DEP_U", "0.002"))      # resource used per presynaptic spike
DEP_TAU = float(os.environ.get("ZF_DEP_TAU", "20.0"))   # seconds to recover
PLASTIC_GROUPS = ("retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right")


def load_graph(graph_path, groups_path):
    """Load a graph. A synthetic one is regenerated from its recipe in memory
    (a pure function of n and seed, seconds to build, no 450 MB file); a real
    EM export is read from the npz build_graph wrote."""
    meta_path = Path(graph_path).with_name("graph.meta.json")
    recipe_path = Path(graph_path).with_name("graph.recipe.json")
    if recipe_path.exists():
        import build_graph

        r = json.loads(recipe_path.read_text())
        ids, coords, types, pre, post, w, groups = build_graph.synthetic_graph(
            r["n"], r.get("seed", 7), r.get("target_fanin"))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("label", "synthetic graph")
        meta.setdefault("neurons", int(len(ids)))
        meta.setdefault("synapses", int(len(pre)))
        return {"meta": meta, "root_ids": ids, "coords": coords, "types": types,
                "pre": pre, "post": post, "w": w,
                "groups": {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}}
    groups_path = Path(groups_path)
    groups = json.loads(groups_path.read_text()) if groups_path.exists() else {}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # a real EM export is a directory of .npy files next to where graph.npz
    # would have been. 39M edges is ~450 MB; opened with mmap the OS pages in
    # what the sim touches and nothing else, and because build_graph already
    # sorted them by `pre` and wrote the CSR indptr, startup does no sorting.
    gdir = Path(graph_path).with_name("graph")
    manifest_path = gdir / "graph.manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        arr = {k: np.load(gdir / f"{k}.npy", mmap_mode="r")
               for k in ("root_ids", "coords", "types", "pre", "post", "w", "indptr")}
        meta.setdefault("label", "Fish1 slice")
        meta.setdefault("neurons", int(man["neurons"]))
        meta.setdefault("synapses", int(man["edges"]))
        return {"meta": meta, "w_absmax": man.get("w_absmax"),
                "indptr": arr["indptr"], **{k: arr[k] for k in
                                            ("root_ids", "coords", "types", "pre", "post", "w")},
                "groups": {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}}

    npz = np.load(graph_path)
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


def to_csr(n, pre, post, w, indptr=None):
    """Outgoing synapses grouped by presynaptic neuron.

    If build_graph already sorted the edges by `pre` and wrote the indptr, this
    is a no-op: the argsort and the two fancy-index copies below are what pull a
    39M-edge memmapped graph into RAM at every startup."""
    if indptr is not None:
        return (np.asarray(indptr, dtype=np.int64),
                np.asarray(post, dtype=np.int32), np.asarray(w, dtype=np.float32))
    order = np.argsort(pre, kind="stable")
    counts = np.bincount(pre, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, post[order].astype(np.int32), w[order].astype(np.float32)


# ---------------------------------------------------------------------------
# kernels (numba when present; plain python is only used by the fallback)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _integrate(v, isyn, refract, hit_p, d, plastic, fired_buf, dt, rest, thresh,
               tau, tau_syn, tau_ref, epsp, dep_tau):
    """One dt for every neuron: leak, synaptic kick, Poisson sensory hit,
    refractory, threshold. Writes the fired indices into fired_buf, returns
    how many. Also recovers the depression resource of plastic neurons."""
    n = v.shape[0]
    k_leak = dt / tau
    k_syn = dt / tau_syn
    k_dep = dt / dep_tau
    m = 0
    for i in range(n):
        vi = v[i] + (rest - v[i]) * k_leak
        kick = isyn[i] * k_syn
        vi += kick
        isyn[i] -= kick
        if hit_p[i] > 0.0 and np.random.random() < hit_p[i]:
            vi += epsp
        r = refract[i] - dt
        if r > 0.0:
            vi = rest
            refract[i] = r
        elif vi >= thresh:
            vi = rest
            refract[i] = tau_ref
            fired_buf[m] = i
            m += 1
        else:
            refract[i] = r
        v[i] = vi
        if plastic[i]:
            d[i] += (1.0 - d[i]) * k_dep
    return m


@njit(cache=True)
def _propagate(indptr, targets, weights, d, plastic, src, m, isyn, dep_u):
    """Event-driven: only the neurons that fired touch their outgoing synapses.
    A plastic (sensory) neuron's synapses deliver w * d and use up U of d."""
    for k in range(m):
        s = src[k]
        ds = d[s]
        for j in range(indptr[s], indptr[s + 1]):
            isyn[targets[j]] += weights[j] * ds
        if plastic[s]:
            d[s] = ds * (1.0 - dep_u)


class FishSim:
    def __init__(self, graph, seed=11):
        self.n = len(graph["coords"])
        self.root_ids = graph["root_ids"]
        self.coords = graph["coords"]
        self.groups = graph.get("groups", {})
        self.meta = graph.get("meta", {})
        self.rng = np.random.default_rng(seed)
        if HAVE_NUMBA:
            _seed(seed)

        # scale weights so typical EPSPs stay in a sane mV band.
        # The fallback scale only needs max|w|, and build_graph puts that in the
        # manifest — computing it here meant a float64 copy plus an abs() of the
        # whole edge table, 600 MB of churn at 39M edges every time a process
        # started.
        scale = float(self.meta.get("weight_scale", 0.0))
        if not scale:
            wmax = graph.get("w_absmax")
            if wmax is None:
                wmax = float(np.abs(graph["w"]).max()) if len(graph["w"]) else 0.0
            scale = EPSP / min(6.0, wmax) if wmax > 0 else 1.0
        self.weight_scale = scale
        indptr = graph.get("indptr")
        self.indptr, self.targets, self.weights = to_csr(
            self.n,
            None if indptr is not None else np.asarray(graph["pre"], dtype=np.int64),
            graph["post"] if indptr is not None else np.asarray(graph["post"], dtype=np.int64),
            (np.asarray(graph["w"], dtype=np.float32) * np.float32(scale)).astype(np.float32),
            indptr=indptr)

        self.v = np.full(self.n, REST_V, dtype=np.float64)
        self.isyn = np.zeros(self.n, dtype=np.float64)
        self.refract = np.zeros(self.n, dtype=np.float64)
        self.spike_counts = np.zeros(self.n, dtype=np.int64)
        self.last_spike = np.full(self.n, -10**9, dtype=np.int64)  # step index of the last spike
        self.step_i = 0
        self.drive = np.zeros(self.n, dtype=np.float64)  # sensory rate, Hz
        self.hit_p = np.zeros(self.n, dtype=np.float64)  # per-step Poisson hit probability
        self.t = 0.0
        self._fired_buf = np.zeros(self.n, dtype=np.int64)

        # habituation lives on the sensory neurons' outgoing synapses
        self.plastic = np.zeros(self.n, dtype=np.bool_)
        for g in self.meta.get("plastic_groups", PLASTIC_GROUPS):
            idx = self.groups.get(g)
            if idx is not None and len(idx):
                self.plastic[np.asarray(idx, dtype=np.int64)] = True
        self.d = np.ones(self.n, dtype=np.float64)

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
                # neuron count for that group (the graphs differ in size)
                rates = np.interp(
                    np.linspace(0, 1, idx.size),
                    np.linspace(0, 1, rates.size), rates)
        self.drive[idx] = rates
        self.hit_p[idx] = 1.0 - np.exp(-rates * DT)

    def clear_drive(self):
        self.drive[:] = 0.0
        self.hit_p[:] = 0.0

    # -- one discrete step ----------------------------------------------
    def _step(self, dt=DT):
        if HAVE_NUMBA:
            m = _integrate(self.v, self.isyn, self.refract, self.hit_p, self.d, self.plastic,
                           self._fired_buf, dt, REST_V, THRESH_V, TAU, TAU_SYN, TAU_REF, EPSP, DEP_TAU)
            self.step_i += 1
            if m:
                src = self._fired_buf[:m]
                self.spike_counts[src] += 1
                self.last_spike[src] = self.step_i
                _propagate(self.indptr, self.targets, self.weights, self.d, self.plastic,
                           src, m, self.isyn, DEP_U)
            return m
        # ---- numpy fallback: the same maths, vectorised ----
        self.v += (REST_V - self.v) * (dt / TAU)
        kick = self.isyn * (dt / TAU_SYN)
        self.v += kick
        self.isyn -= kick
        hits = self.rng.random(self.n) < self.hit_p
        self.v[hits] += EPSP
        self.refract -= dt
        blocked = self.refract > 0
        self.v[blocked] = REST_V
        fired = (self.v >= THRESH_V) & ~blocked
        self.d[self.plastic] += (1.0 - self.d[self.plastic]) * (dt / DEP_TAU)
        self.step_i += 1
        src = np.flatnonzero(fired)
        if src.size:
            self.spike_counts[src] += 1
            self.last_spike[src] = self.step_i
            self.v[src] = REST_V
            self.refract[src] = TAU_REF
            starts, ends = self.indptr[src], self.indptr[src + 1]
            lens = ends - starts
            tot = int(lens.sum())
            if tot:
                offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
                idx = np.arange(tot) + offs
                wd = self.weights[idx] * np.repeat(self.d[src], lens)
                self.isyn += np.bincount(self.targets[idx], weights=wd, minlength=self.n)
            ps = src[self.plastic[src]]
            self.d[ps] *= (1.0 - DEP_U)
        return int(src.size)

    # -- full epoch -------------------------------------------------------
    def run(self, n_steps=400, dt=DT, tick=None, tick_every=0):
        """Run n_steps of biological time. Returns detail dict (see stream()).

        `tick` is called every `tick_every` steps, between slices - the roamer
        uses it to keep its camera running at video rate while the brain
        thinks, since a whole-brain step takes most of a second. The rate
        window is still the full n_steps: slicing changes when other work
        happens, not what the brain computes."""
        if not tick or tick_every <= 0:
            for _ in range(n_steps):
                self._step(dt)
        else:
            done = 0
            while done < n_steps:
                chunk = min(tick_every, n_steps - done)
                for _ in range(chunk):
                    self._step(dt)
                done += chunk
                if done < n_steps:
                    tick()
        self.t += n_steps * dt
        return self.stream(dt, n_steps)

    # -- readout (the panels the live site shows) -------------------------
    def stream(self, dt=DT, window_steps=400):
        window_spikes = self.spike_counts.copy()
        self.spike_counts[:] = 0  # per-window counters

        def rate(name):
            """Mean firing rate per neuron in the population, Hz."""
            idx = self.groups.get(name)
            if idx is None or len(idx) == 0:
                return 0.0
            return float(window_spikes[idx].sum()) / (len(idx) * window_steps * dt)

        rates = {g: rate(g) for g in self.groups}
        recent = self.last_spike >= self.step_i - FLASH_STEPS
        return {
            # packed bits, one per neuron — 23 KB at whole-brain scale, so it
            # travels as bytes on /firing.bin, never inside the heartbeat JSON
            "firing_bits": np.packbits(recent).tobytes(),
            "firing_recent": int(recent.sum()),
            "t": round(self.t, 3),
            "neurons": int(self.n),
            "spikes_per_sec": int(window_spikes.sum() / (window_steps * dt)),
            "mean_membrane_mv": round(float(self.v.mean()), 2),
            "habituation": round(float(1.0 - self.d[self.plastic].mean()), 4) if self.plastic.any() else 0.0,
            "visual_spikes": int(window_spikes[self.groups.get("retina", [])].sum())
            if "retina" in self.groups else 0,
            "motor_spikes": int(window_spikes[self.groups.get("spinal", [])].sum())
            if "spinal" in self.groups else 0,
            "rates_hz": {k: round(v, 2) for k, v in rates.items()},
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
        nmlf_span = max(MIN_SPAN, SPAN_FACTOR * nmlf_rest)
        vspn_span = max(MIN_SPAN, SPAN_FACTOR * vspn_rest)
        out = {"dx": 0.0, "dy": 0.0, "scroll": 0.0, "turn": 0.0, "escape": False}

        # a fish steers by asymmetry: the difference between opposed DSGC
        # pairs, not the loudest one — a balanced field is no steering at all
        up, dn = rates.get("dsgc_up", 0.0), rates.get("dsgc_down", 0.0)
        lf, rt = rates.get("dsgc_left", 0.0), rates.get("dsgc_right", 0.0)
        out["dx"] = float(np.clip((rt - lf) / max(rt + lf, 1.0), -1.0, 1.0))
        out["dy"] = float(np.clip((dn - up) / max(dn + up, 1.0), -1.0, 1.0))
        strongest = max(up, dn, lf, rt)

        # nMLF bout gate -> a scroll burst when the population rises above rest
        out["scroll"] = float(np.clip((rates.get("nmlf", 0.0) - nmlf_rest) / nmlf_span, 0.0, 1.0))

        # vSPN above rest -> a turn, signed against the DSGC winner
        out["turn"] = float(np.clip((rates.get("vspn", 0.0) - vspn_rest) / vspn_span, 0.0, 1.0)) * (-1 if strongest else 1)

        # Mauthner: an escape is all-or-nothing and fast — a startle is a burst,
        # not the odd coincidence spike a cell with hundreds of inputs throws.
        # Burst = rise above the fish's own recent baseline by ESCAPE_BURST×,
        # AND above the absolute floor ESCAPE_HZ.  A hot page that keeps M-cell
        # high constantly does not count — that is a habit, not a startle.
        m_rest = rest.get("mauthner", MAUTHNER_REST)
        m_rate = rates.get("mauthner", 0.0)
        out["escape"] = m_rate > ESCAPE_HZ and m_rate > m_rest * ESCAPE_BURST
        return out


NEED = ["retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right",
        "nmlf", "vspn", "mauthner", "spinal"]


def smoke(graph_path="build/graph.npz", groups_path="build/groups.json"):
    """Checks that hold for any graph we run — the toy one, the synthetic one,
    and the real one when it lands.

    1. under a plain retina drive every readout population fires (> 0 Hz) and
       none saturates (< 350 Hz; the 2 ms refractory ceiling is 500)
    2. a still page makes the fish neither bout, turn nor escape
    3. a rightward DSGC drive steers right
    4. Mauthner sits below ESCAPE_HZ at rest and a whole-field flash lifts it
       at least 5x — the startle is a burst, not a habit
    5. habituation: the same flash held for seconds startles it less and less
    """
    graph = load_graph(graph_path, groups_path)
    label = graph["meta"].get("label", "graph")
    real = graph["meta"].get("source") != "smoke"
    ds = ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right")

    def page_drive(sim, lum=40.0, flow=40.0):
        """What Retina.rates hands set_drive on an ordinary page: a luminance
        drive to the retina and an optic-flow drive to each motion channel."""
        sim.set_drive(graph["groups"]["retina"], lum)
        for g in ds:
            sim.set_drive(graph["groups"][g], flow)

    sim = FishSim(graph)
    page_drive(sim, flow=0.0)          # a page the fish is not moving
    for _ in range(12):
        d1 = sim.run(400)  # let the recurrent pools and the depression settle
    r1 = d1["rates_hz"]
    # the Mauthner cell is meant to be silent between escapes — a larva's M-cell
    # fires once for a startle, not continuously; every other population lives
    dead = [g for g in NEED if g != "mauthner" and r1.get(g, 0.0) <= 0.0]
    hot = [g for g in NEED if r1.get(g, 0.0) >= 350.0]
    print(f"[{label}] rest rates:", {g: r1[g] for g in NEED}, f"habituation {d1['habituation']:.3f}")
    if real:
        assert not dead, f"silent populations under retina drive: {dead}"
    assert not hot, f"saturated populations: {hot}"
    assert r1.get("mauthner", 0.0) < ESCAPE_HZ, "Mauthner startles at rest"
    rest = sim.decode(r1)
    assert not rest["escape"] and rest["scroll"] < 0.45 and abs(rest["turn"]) < 0.5, \
        f"a still page should not make the fish bout, turn or escape: {rest}"
    print("rest decode OK:", {k: round(v, 2) if isinstance(v, float) else v for k, v in rest.items()})

    sim2 = FishSim(graph, seed=5)
    page_drive(sim2, flow=0.0)
    sim2.set_drive(graph["groups"]["dsgc_right"], 120.0)
    dec = sim2.decode(sim2.run(400)["rates_hz"])
    assert dec["dx"] > 0, f"rightward input should steer right, got {dec}"
    print("steer OK:", {k: round(v, 2) if isinstance(v, float) else v for k, v in dec.items()})

    def flash(sim):
        # a flash = a luminance jump (retina) plus one frame of whole-field motion
        page_drive(sim, 400.0, 200.0)
        m = sim.run(200)["rates_hz"].get("mauthner", 0.0)
        page_drive(sim, flow=0.0)
        return m

    sim3 = FishSim(graph, seed=3)
    page_drive(sim3, flow=0.0)
    for _ in range(12):
        base = sim3.run(400)["rates_hz"].get("mauthner", 0.0)
    train = []
    for _ in range(10):  # ten flashes, 0.8 s apart — the classic startle-habituation assay
        train.append(flash(sim3))
        for _ in range(4):
            sim3.run(400)
    print(f"startle: mauthner {base:.1f} Hz at rest -> {train[0]:.1f} Hz on a flash (escape at {ESCAPE_HZ})")
    print("habituation: flash train ->", " ".join(f"{m:.0f}" for m in train), "Hz")
    if real:
        assert train[0] >= max(5 * base, ESCAPE_HZ), "a whole-field flash should startle the Mauthner cell"
        assert train[-1] < 0.6 * train[0], "ten flashes should habituate the startle"
    print("smoke OK")


def bench(n, deg=167, steps=200, seed=0):
    """In-memory timing at whole-brain scale: n neurons, deg synapses each."""
    rng = np.random.default_rng(seed)
    e = n * deg
    graph = {"root_ids": np.arange(n), "coords": rng.normal(0, 1, (n, 3)).astype(np.float32),
             "types": np.array(["unknown"] * 1), "pre": np.repeat(np.arange(n, dtype=np.int64), deg),
             "post": rng.integers(0, n, e, dtype=np.int64),
             "w": (rng.choice([-1.0, 1.0], e) * rng.uniform(1, 6, e)).astype(np.float32),
             "groups": {"retina": np.arange(0, n // 10)},
             # a random graph has no E/I structure; pin the scale so the mean
             # synapse is ~0.6 mV and the bench runs at a realistic 10-30 Hz
             "meta": {"source": "bench", "weight_scale": 0.15}}
    t0 = time.time()
    sim = FishSim(graph)
    print(f"{n:,} neurons / {e:,} synapses: graph + CSR in {time.time()-t0:.1f}s, numba={HAVE_NUMBA}")
    sim.set_drive(graph["groups"]["retina"], 45.0)
    sim.run(20)  # warm the JIT
    t0 = time.time()
    d = sim.run(steps)
    el = time.time() - t0
    print(f"{steps} steps in {el:.2f}s -> {el/steps*1000:.2f} ms/step, {d['spikes_per_sec']/n:.0f} Hz/neuron, "
          f"400-step frame = {el/steps*400:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="build/graph.npz")
    ap.add_argument("--groups", default="build/groups.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bench", type=int, metavar="N", help="time N neurons x 167 synapses in memory")
    args = ap.parse_args()
    if args.bench:
        bench(args.bench)
        return

    graph = load_graph(args.graph, args.groups)
    sim = FishSim(graph)
    sim.set_drive(graph["groups"].get("retina", []), 100.0)
    for side in ("right", "up"):
        sim.set_drive(graph["groups"].get(f"dsgc_{side}", []), 100.0)
    d = sim.run(args.steps)
    print(json.dumps({k: v for k, v in d.items() if k != "firing_bits"}, indent=2))
    print("decode:", sim.decode(d["rates_hz"]))
    smoke(args.graph, args.groups)


if __name__ == "__main__":
    main()
