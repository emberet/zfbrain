"""Build the connectome graph the sim runs on: data/raw/*.csv -> build/graph.npz.

Larval zebrafish whole-brain connectome (Fish1, CC-BY research release).
We keep the circuits the release paper already dissected, drop pairs with
fewer than 3 synapses, and sign each edge from the molecular type of the
presynaptic neuron (vglut2a = excitatory, gad1b = inhibitory) exactly the way
the fly project signs from predicted neurotransmitters.

Inputs (written by fetch_cave.py, or by hand):
  data/raw/neurons.csv       id,x,y,z,type            type in {glu,gaba,unknown}
  data/raw/synapses.csv      pre,post,count[,sign]    sign +1/-1 or inferred
  data/raw/groups.csv        group,id                 readout wiring (optional)

Output:
  build/graph.npz   pre, post, w, coords, types, root_ids
  build/groups.json {group: [neuron index, ...]}

Usage:
  python build_graph.py               # real data
  python build_graph.py --smoke       # 60-neuron toy graph for the unit test
  python build_graph.py --synthetic   # 7,000-neuron fish-shaped synthetic brain,
                                      # every population wired (the stand-in
                                      # until Fish1's proofread export lands)
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw")
BUILD = Path("build")
MIN_SYNAPSES = 3


def load_neurons():
    path = DATA / "neurons.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run fetch_cave.py first, or use --smoke.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["id"])
    df["type"] = df["type"].astype(str).str.lower()
    df["type"] = df["type"].where(df["type"].isin(["glu", "gaba"]), "unknown")
    ids = df["id"].astype(np.int64).to_numpy()
    coords = df[["x", "y", "z"]].to_numpy(np.float32)
    types = df["type"].to_numpy()
    index = {rid: i for i, rid in enumerate(ids)}
    return ids, coords, types, index


def load_edges(index, types_by_id):
    path = DATA / "synapses.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run fetch_cave.py first, or use --smoke.")
    df = pd.read_csv(path)
    df = df[df["pre"].isin(index.keys()) & df["post"].isin(index.keys())]
    if "count" in df:
        df = df[df["count"] >= MIN_SYNAPSES]
    pre = df["pre"].map(index).astype(np.int32).to_numpy()
    post = df["post"].map(index).astype(np.int32).to_numpy()
    if "sign" in df.columns:
        sign = df["sign"].to_numpy(np.float32)
    else:
        sign = np.where(
            df["pre"].map(lambda rid: types_by_id.get(rid) == "glu").to_numpy(), 1.0, -1.0
        )
    w = (df["count"].to_numpy() if "count" in df else np.ones(len(df))) * sign
    return pre.astype(np.int32), post.astype(np.int32), w.astype(np.float32)


def load_groups(ids, index):
    path = DATA / "groups.csv"
    groups = {}
    if path.exists():
        gdf = pd.read_csv(path)
        for _, row in gdf.iterrows():
            name, rid = str(row["group"]), int(row["id"])
            if rid in index:
                groups.setdefault(name, []).append(index[rid])
    return groups


def smoke_graph(seed=7):
    """Tiny deterministic graph so the sim is runnable before real data."""
    rng = np.random.default_rng(seed)
    groups = {
        "retina": list(range(16)),
        "dsgc_up": list(range(16, 20)),
        "dsgc_down": list(range(20, 24)),
        "dsgc_left": list(range(24, 28)),
        "dsgc_right": list(range(28, 32)),
        "nmlf": list(range(32, 42)),
        "vspn": list(range(42, 46)),
        "mauthner": [46],
        "spinal": list(range(47, 60)),
    }
    n = 60
    ids = np.arange(n, dtype=np.int64) + 100000
    coords = rng.normal(0, 20, (n, 3)).astype(np.float32)
    types = np.array(["unknown"] * n)
    edges = []
    ret, up = 16, 48
    for li in range(8):
        for dgi in range(16, 32):
            edges.append((li, dgi, 4, -1.0 if rng.random() < 0.4 else 1.0))
    for gi in (32, 36, 40, 44):  # dsgc -> nmlf/vspn
        for t in range(gi, gi + 2):
            edges.append((gi, t, 3, 1.0))
    for v in range(42, 46):      # vspn -> spinal (turn)
        for s in range(47, 55):
            edges.append((v, s, 5, -1.0 if (v + s) % 2 else 1.0))
    edges.append((0, 46, 3, -1.0))  # retina -> mauthner (startle path)
    edges.append((46, 55, 6, 1.0))  # mauthner -> contralateral spinal
    pre = np.array([e[0] for e in edges], np.int32)
    post = np.array([e[1] for e in edges], np.int32)
    w = np.array([e[3] for e in edges], np.float32)
    return ids, coords, types, pre, post, w, groups


# ---------------------------------------------------------------------------
# synthetic fish-shaped brain — the honest stand-in until Fish1 is exported
# ---------------------------------------------------------------------------
def _profile(x):
    """Larva lateral profile, head left: x 0..1 along the fish, y up-negative.
    Same function the site's hero canvas uses, so neuron = dot."""
    u = min(1.0, x / 0.8)
    s = math.sin(math.pi * u ** 0.72) ** 0.75
    return (-(0.012 + 0.092 * s) * (1 - 0.5 * u * u),
            (0.012 + 0.108 * s) * (1 - 0.6 * u * u))


def _in_fish(x, y):
    """0 outside, 1 body, 2 fin, 3 caudal fin — ported 1:1 from the page."""
    if x < 0 or x > 1:
        return 0
    if x <= 0.8:
        top, bottom = _profile(x)
        if top <= y <= bottom:
            return 1
        if 0.50 < x < 0.66 and y < top and y > top - 0.07 * math.sin((x - 0.50) / 0.16 * math.pi) ** 0.5 * ((x - 0.50) / 0.05 if x < 0.55 else 1):
            return 2  # dorsal fin
        if 0.60 < x < 0.78 and y > bottom and y < bottom + 0.055 * math.sin((x - 0.60) / 0.18 * math.pi) ** 0.6:
            return 2  # anal fin
        if 0.20 < x < 0.34 and y > bottom * 0.55 and y < bottom * 0.55 + 0.05 * (1 - abs((x - 0.27) / 0.07)):
            return 2  # pectoral fin
        if 0.40 < x < 0.48 and y > bottom and y < bottom + 0.028 * (1 - abs((x - 0.44) / 0.04)):
            return 2  # pelvic fin
        return 0
    v = (x - 0.8) / 0.2
    half = 0.03 + 0.085 * v ** 0.8
    notch = (v - 0.55) / 0.45 * 0.055 if v > 0.55 else 0.0
    return 3 if notch <= abs(y) <= half else 0


def fish_points(n=7000, seed=23):
    """The page's dot sampler (same LCG, same rules): n points inside the larva."""
    s = seed

    def rnd():
        nonlocal s
        s = (s * 16807) % 2147483647
        return s / 2147483647

    pts = []
    while len(pts) < n:
        x = rnd()
        y = (rnd() - 0.5) * 0.36
        kind = _in_fish(x, y)
        if not kind:
            continue
        if math.hypot(x - 0.105, y + 0.02) < 0.017:
            continue  # pupil: empty
        if kind == 2 and rnd() < 0.55:
            continue  # fins are sparse
        rnd()  # the page draws a phase value here; keep the sequence identical
        pts.append((x, y, kind))
    return pts


# Fan-in per connection rule, as a share of TARGET_FANIN. The ratios between
# pathways are the tuned ones; the absolute numbers are scaled at build time so
# the whole graph lands on TARGET_FANIN synapses per neuron whatever N is.
SYN = {
    "retina->dsgc":   dict(k=16, count=6, p_exc=0.72),   # sized for a page-like 40 Hz retinal drive
    "dsgc->nmlf":     dict(k=8,  count=5),
    "dsgc->vspn":     dict(k=8,  count=5),
    "dsgc->other":    dict(k=9,  count=5),   # wakes the integrator pool at 208 fan-in
    "other->other":   dict(k=5,  count=3),
    "other->nmlf":    dict(k=3,  count=2),
    "other->vspn":    dict(k=3,  count=2),   # the hindbrain drives the turn too, not only the bout
    "vspn->other":    dict(k=2,  count=2),
    # feedback inhibition: the motor pools drive the hindbrain's GABA cells,
    # which brake them back. Without this loop the graph has one narrow
    # operating point between silence and runaway; with it, it self-limits.
    "motor->inh":     dict(k=4,  count=3),
    "inh->motor":     dict(k=2,  count=3),
    "dsgc->mauthner": dict(k=25, count=1),   # below the refractory ceiling, so habituation has room to show
    "nmlf->mauthner": dict(k=10, count=1),   # the M-cell's brake: silent at rest, ~390 Hz on a flash
    "nmlf->spinal":   dict(k=8,  count=4),   # the cord needs real drive from a 10 Hz nMLF
    "other->spinal":  dict(k=4,  count=4),   # reticulospinal: the hindbrain drives the cord too
    "vspn->spinal":   dict(k=5,  count=4),
    "mauthner->spinal": dict(k=500, count=6),   # out-degree, not fan-in: the C-start
    "spinal->spinal": dict(k=3,  count=3),      # chain length down the cord
}
# Share of the graph each population takes. The Mauthner cell is the exception:
# a larva has exactly one pair however big the graph is.
FRACTIONS = dict(retina=0.2571, dsgc=0.1429, nmlf=0.05, vspn=0.05, other=0.20)  # spinal = the rest
MAUTHNER = 2
GABA_FRACTION_OTHER = 0.30
# Inhibitory synapses are stronger than excitatory ones (they land closer to the
# soma). Without that asymmetry a randomly wired E/I graph has no stable
# operating point: it is silent, or it runs away. INH_GAIN is the ratio.
INH_GAIN = float(os.environ.get("ZF_INH_GAIN", "2.0"))
# The real larva: the Harvard/Google 7 dpf reconstruction counts 187,053 cell
# bodies and ~39M synapses, i.e. ~208 synapses per neuron. The synthetic graph
# matches those numbers; the wiring is ours, not theirs.
REAL_NEURONS = 187053
TARGET_FANIN = 208


def _sizes(n):
    """Population sizes for a graph of n neurons; spinal takes the remainder."""
    sizes = {k: int(round(v * n)) for k, v in FRACTIONS.items()}
    sizes["mauthner"] = MAUTHNER
    used = sum(sizes.values())
    sizes["spinal"] = n - used
    if sizes["spinal"] < 1:
        raise SystemExit(f"n={n} is too small for the population layout")
    return sizes


def _fanin_multiplier(sizes, target=None):
    """How much to scale every k so the graph averages `target` synapses per
    neuron. Computed from the rule table, so the ratios between pathways stay
    exactly as tuned."""
    n = sum(sizes.values())
    dst = {"retina->dsgc": sizes["dsgc"], "dsgc->nmlf": sizes["nmlf"], "dsgc->vspn": sizes["vspn"],
           "dsgc->other": sizes["other"], "other->other": sizes["other"], "other->nmlf": sizes["nmlf"],
           "other->vspn": sizes["vspn"], "vspn->other": sizes["other"],
           "motor->inh": int(sizes["other"] * GABA_FRACTION_OTHER),
           "inh->motor": sizes["nmlf"] + sizes["vspn"], "dsgc->mauthner": sizes["mauthner"],
           "nmlf->mauthner": sizes["mauthner"], "nmlf->spinal": sizes["spinal"],
           "vspn->spinal": sizes["spinal"], "other->spinal": sizes["spinal"],
           "mauthner->spinal": sizes["mauthner"],
           "spinal->spinal": sizes["spinal"]}
    base = sum(SYN[rule]["k"] * count for rule, count in dst.items()) / n
    return (TARGET_FANIN if target is None else target) / base


def synthetic_graph(n=REAL_NEURONS, seed=7, target_fanin=None):
    """A fish-shaped, fully wired synthetic connectome at the larva's own scale.

    Populations are laid out head->tail by rank on x (retina in the eye, DSGCs
    in the tectum, nMLF / vSPN / the Mauthner pair / an integrator pool in the
    hindbrain, spinal cord down the body and tail). Every dot on the site is
    one of these neurons. Edges are built as whole numpy arrays per rule - at
    39M synapses there is no per-synapse python left anywhere."""
    rng = np.random.default_rng(seed)
    sizes = _sizes(n)
    mult = _fanin_multiplier(sizes, target_fanin)
    K = {rule: max(1, int(round(v["k"] * mult))) for rule, v in SYN.items()}

    pts = fish_points(n)
    coords = np.empty((n, 3), np.float32)
    coords[:, 0] = [p[0] for p in pts]
    coords[:, 1] = [p[1] for p in pts]
    coords[:, 2] = 0.0
    order = np.argsort(coords[:, 0], kind="stable")     # head -> tail

    groups = {}
    i = 0
    for name in ("retina", "dsgc", "nmlf", "vspn", "mauthner", "other"):
        groups[name] = order[i:i + sizes[name]]
        i += sizes[name]
    groups["spinal"] = order[i:]

    # retina: index order (y, x) so the 12x18 retina grid resamples onto it retinotopically
    r = groups["retina"]
    groups["retina"] = r[np.lexsort((coords[r, 0], coords[r, 1]))]
    # DSGC directions by quadrant around the tectum centroid
    d = groups["dsgc"]
    cx, cy = coords[d, 0].mean(), coords[d, 1].mean()
    up, dn = coords[d, 1] < cy, coords[d, 1] >= cy
    lf, rt = coords[d, 0] < cx, coords[d, 0] >= cx
    groups["dsgc_up"], groups["dsgc_down"] = d[up & lf], d[dn & lf]
    groups["dsgc_left"], groups["dsgc_right"] = d[up & rt], d[dn & rt]
    del groups["dsgc"]
    # the Mauthner pair sits on the midline, one per side
    m = groups["mauthner"]
    groups["mauthner"] = m[np.argsort(coords[m, 1])][[0, -1]] if len(m) > 1 else m

    is_gaba = np.zeros(n, dtype=bool)
    other = groups["other"]
    gaba = rng.random(len(other)) < GABA_FRACTION_OTHER
    is_gaba[other[gaba]] = True

    chunks = []

    def connect(src, dst, k, count, sign=None, p_exc=None, interneuron=False):
        """Every dst neuron gets k random src inputs, as one array of edges.
        `interneuron` marks a projection made by real inhibitory cells, whose
        synapses carry INH_GAIN times the weight of an excitatory one."""
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.size == 0 or dst.size == 0:
            return
        kk = int(min(k, src.size))
        pre_i = src[rng.integers(0, src.size, dst.size * kk)].astype(np.int32)
        post_i = np.repeat(dst.astype(np.int32), kk)
        if sign is not None:
            sg = np.full(pre_i.size, sign, np.float32)
        elif p_exc is not None:
            sg = np.where(rng.random(pre_i.size) < p_exc, 1.0, -1.0).astype(np.float32)
        else:                                   # sign from the presynaptic cell's type
            sg = np.where(is_gaba[pre_i], -1.0, 1.0).astype(np.float32)
        if interneuron:
            sg = np.where(sg < 0, sg * INH_GAIN, sg)
        chunks.append((pre_i, post_i, (count * sg).astype(np.float32)))

    dsgc = np.concatenate([groups[g] for g in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right")])
    c = SYN
    connect(groups["retina"], dsgc, K["retina->dsgc"], c["retina->dsgc"]["count"], p_exc=c["retina->dsgc"]["p_exc"])
    connect(np.concatenate([groups["dsgc_up"], groups["dsgc_down"]]), groups["nmlf"],
            K["dsgc->nmlf"], c["dsgc->nmlf"]["count"], sign=1.0)
    connect(np.concatenate([groups["dsgc_left"], groups["dsgc_right"]]), groups["vspn"],
            K["dsgc->vspn"], c["dsgc->vspn"]["count"], sign=1.0)
    connect(dsgc, other, K["dsgc->other"], c["dsgc->other"]["count"], sign=1.0)
    connect(other, other, K["other->other"], c["other->other"]["count"], interneuron=True)  # sign from the pre cell
    connect(other[~gaba], groups["nmlf"], K["other->nmlf"], c["other->nmlf"]["count"], sign=1.0)
    connect(other[~gaba], groups["vspn"], K["other->vspn"], c["other->vspn"]["count"], sign=1.0)
    connect(groups["vspn"], other, K["vspn->other"], c["vspn->other"]["count"], sign=-1.0, interneuron=True)
    # the brake: nMLF + vSPN excite the hindbrain's inhibitory cells, which
    # inhibit them straight back
    motor = np.concatenate([groups["nmlf"], groups["vspn"]])
    inh = other[gaba]
    connect(motor, inh, K["motor->inh"], c["motor->inh"]["count"], sign=1.0)
    connect(inh, motor, K["inh->motor"], c["inh->motor"]["count"], sign=-1.0, interneuron=True)
    connect(dsgc, groups["mauthner"], K["dsgc->mauthner"], c["dsgc->mauthner"]["count"], sign=1.0)
    connect(groups["nmlf"], groups["mauthner"], K["nmlf->mauthner"], c["nmlf->mauthner"]["count"], sign=-1.0)
    connect(groups["nmlf"], groups["spinal"], K["nmlf->spinal"], c["nmlf->spinal"]["count"], sign=1.0)
    connect(other[~gaba], groups["spinal"], K["other->spinal"], c["other->spinal"]["count"], sign=1.0)

    # vSPN -> spinal: excites its own side of the body, inhibits the other (the turn)
    sp = groups["spinal"]
    vs = groups["vspn"]
    kk = K["vspn->spinal"]
    pre_i = vs[rng.integers(0, vs.size, sp.size * kk)].astype(np.int32)
    post_i = np.repeat(sp.astype(np.int32), kk)
    same = (coords[pre_i, 1] < 0) == (coords[post_i, 1] < 0)
    chunks.append((pre_i, post_i, (c["vspn->spinal"]["count"] * np.where(same, 1.0, -1.0)).astype(np.float32)))

    # Mauthner -> contralateral spinal, strong: the C-start
    for mj in groups["mauthner"]:
        contra = sp[(coords[sp, 1] < 0) != (coords[mj, 1] < 0)]
        if contra.size == 0:
            continue
        tgt = rng.choice(contra, size=int(min(K["mauthner->spinal"], contra.size)), replace=False)
        chunks.append((np.full(tgt.size, mj, np.int32), tgt.astype(np.int32),
                       np.full(tgt.size, c["mauthner->spinal"]["count"], np.float32)))

    # spinal chain head -> tail: the bout travels down the cord
    sp_sorted = sp[np.argsort(coords[sp, 0])].astype(np.int32)
    steps = np.arange(1, K["spinal->spinal"] + 1)
    src_i = np.repeat(np.arange(sp_sorted.size, dtype=np.int64), steps.size)
    dst_i = src_i + np.tile(steps, sp_sorted.size)
    keep = dst_i < sp_sorted.size
    chunks.append((sp_sorted[src_i[keep]], sp_sorted[dst_i[keep]],
                   np.full(int(keep.sum()), c["spinal->spinal"]["count"], np.float32)))

    pre = np.concatenate([ch[0] for ch in chunks])
    post = np.concatenate([ch[1] for ch in chunks])
    w = np.concatenate([ch[2] for ch in chunks])
    chunks.clear()

    types = np.where(is_gaba, "gaba", "glu")
    ids = np.arange(n, dtype=np.int64) + 200000
    groups = {k: [int(v) for v in vals] for k, vals in groups.items()}
    return ids, coords, types, pre, post, w, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="build the 60-neuron toy graph")
    ap.add_argument("--synthetic", nargs="?", const=REAL_NEURONS, type=int, metavar="N",
                    help=f"build the fish-shaped synthetic brain (default {REAL_NEURONS:,}, the larva's own count)")
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    recipe = BUILD / "graph.recipe.json"
    if recipe.exists():
        recipe.unlink()
    if args.smoke:
        ids, coords, types, pre, post, w, groups = smoke_graph()
    elif args.synthetic:
        ids, coords, types, pre, post, w, groups = synthetic_graph(args.synthetic)
    else:
        ids, coords, types, index = load_neurons()
        types_by_id = {rid: t for rid, t in zip(ids, types)}
        pre, post, w = load_edges(index, types_by_id)
        groups = load_groups(ids, index)

    if args.synthetic:
        # the synthetic graph is a pure function of (n, seed): storing 39M edges
        # would be ~450 MB of disk to reproduce something that builds in seconds,
        # so write the recipe and let load_graph regenerate it in memory
        with open(recipe, "w") as f:
            json.dump({"kind": "synthetic", "n": int(args.synthetic), "seed": 7,
                       "target_fanin": TARGET_FANIN}, f, indent=2)
        for stale in (BUILD / "graph.npz", BUILD / "groups.json"):
            if stale.exists():
                stale.unlink()
    else:
        np.savez_compressed(
            BUILD / "graph.npz",
            root_ids=ids,
            coords=coords,
            types=np.array([t.encode() for t in types]),
            pre=pre,
            post=post,
            w=w,
        )
        with open(BUILD / "groups.json", "w") as f:
            json.dump({k: v for k, v in groups.items()}, f)
    # what the live site's header says about this graph — never a guess
    with open(BUILD / "graph.meta.json", "w") as f:
        source = "smoke" if args.smoke else ("synthetic" if args.synthetic else "fish1")
        meta = {"source": source,
                "label": {"smoke": "smoke graph", "synthetic": "synthetic graph"}.get(source, "Fish1 slice"),
                "neurons": int(len(ids)), "synapses": int(len(pre)),
                "fanin": round(len(pre) / max(1, len(ids)), 1),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if source == "synthetic":
            # first guess at the weight scale: the same charge per neuron as the
            # small graph had, spread over this graph's fan-in. calibrate.py
            # refines it until fishsim's assertions pass.
            meta["weight_scale"] = round((4.0 / 6.0) * (8.4 / max(1.0, meta["fanin"])), 6)
        json.dump(meta, f, indent=2)

    print(f"neurons      {len(ids)}")
    print(f"edges (signed){len(pre)}")
    print(f"excitatory   {(w > 0).sum()}")
    print(f"inhibitory   {(w < 0).sum()}")
    print("groups:", ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"-> {recipe if args.synthetic else BUILD/'graph.npz'} + {BUILD/'graph.meta.json'}")


if __name__ == "__main__":
    main()