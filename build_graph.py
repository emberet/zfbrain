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


# synapse counts per connection; FishSim maps count 6 -> EPSP mV, count 1 -> EPSP/6
SYN = {
    "retina->dsgc":   dict(k=10, count=5, p_exc=0.72),
    "dsgc->nmlf":     dict(k=8,  count=4),
    "dsgc->vspn":     dict(k=8,  count=4),
    "dsgc->other":    dict(k=6,  count=4),
    "other->other":   dict(k=5,  count=3),
    "other->nmlf":    dict(k=3,  count=2),
    "vspn->other":    dict(k=2,  count=2),
    "dsgc->mauthner": dict(k=60, count=1),   # tuned: ~8 Hz at rest, ~180 Hz on a whole-field flash
    "nmlf->mauthner": dict(k=20, count=1),   # tuned: ~8 Hz at rest, ~260 Hz on a flash
    "nmlf->spinal":   dict(k=3,  count=4),
    "vspn->spinal":   dict(k=2,  count=4),
    "mauthner->spinal": dict(k=500, count=6),
    "spinal->spinal": dict(k=3,  count=3),
}
SIZES = dict(retina=1800, dsgc=1000, nmlf=350, vspn=350, mauthner=2, other=1400)  # spinal = the rest
GABA_FRACTION_OTHER = 0.30


def synthetic_graph(n=7000, seed=7):
    """A fish-shaped, fully wired synthetic connectome. Populations are laid
    out head->tail by rank on x (retina in the eye, DSGCs in the tectum, nMLF /
    vSPN / the Mauthner pair / an integrator pool in the hindbrain, spinal cord
    down the body and tail). Every dot on the site is one of these neurons."""
    rng = np.random.default_rng(seed)
    pts = fish_points(n)
    coords = np.array([(x, y, 0.0) for x, y, _ in pts], np.float32)
    order = np.argsort(coords[:, 0], kind="stable")     # head -> tail

    groups = {}
    i = 0
    for name in ("retina", "dsgc", "nmlf", "vspn", "mauthner", "other"):
        groups[name] = order[i:i + SIZES[name]]
        i += SIZES[name]
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

    types = np.array(["glu"] * n, dtype=object)
    other = groups["other"]
    gaba = rng.random(len(other)) < GABA_FRACTION_OTHER
    types[other[gaba]] = "gaba"
    types[groups["retina"]] = "glu"

    pre, post, w = [], [], []

    def connect(src, dst, k, count, sign=None, p_exc=None):
        """Every dst neuron gets k random src inputs."""
        src = np.asarray(src)
        if len(src) == 0 or len(dst) == 0:
            return
        picks = rng.choice(src, size=(len(dst), min(k, len(src))), replace=True)
        for j, dj in enumerate(dst):
            for sj in picks[j]:
                if sign is not None:
                    sg = sign
                elif p_exc is not None:
                    sg = 1.0 if rng.random() < p_exc else -1.0
                else:
                    sg = -1.0 if types[sj] == "gaba" else 1.0
                pre.append(sj); post.append(dj); w.append(count * sg)

    dsgc = np.concatenate([groups[g] for g in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right")])
    c = SYN
    connect(groups["retina"], dsgc, c["retina->dsgc"]["k"], c["retina->dsgc"]["count"], p_exc=c["retina->dsgc"]["p_exc"])
    connect(np.concatenate([groups["dsgc_up"], groups["dsgc_down"]]), groups["nmlf"], **c["dsgc->nmlf"], sign=1.0)
    connect(np.concatenate([groups["dsgc_left"], groups["dsgc_right"]]), groups["vspn"], **c["dsgc->vspn"], sign=1.0)
    connect(dsgc, other, **c["dsgc->other"], sign=1.0)
    connect(other, other, **c["other->other"])                      # sign from the pre cell's type
    connect(other[~gaba], groups["nmlf"], **c["other->nmlf"], sign=1.0)
    connect(groups["vspn"], other, **c["vspn->other"], sign=-1.0)
    connect(dsgc, groups["mauthner"], **c["dsgc->mauthner"], sign=1.0)
    connect(groups["nmlf"], groups["mauthner"], **c["nmlf->mauthner"], sign=-1.0)
    connect(groups["nmlf"], groups["spinal"], **c["nmlf->spinal"], sign=1.0)
    # vSPN -> spinal: excites its own side, inhibits the other (the turn)
    sp = groups["spinal"]
    picks = rng.choice(groups["vspn"], size=(len(sp), c["vspn->spinal"]["k"]))
    for j, dj in enumerate(sp):
        for sj in picks[j]:
            same = (coords[sj, 1] < 0) == (coords[dj, 1] < 0)
            pre.append(sj); post.append(dj); w.append(c["vspn->spinal"]["count"] * (1.0 if same else -1.0))
    # Mauthner -> contralateral spinal, strong: the C-start
    for mj in groups["mauthner"]:
        contra = sp[(coords[sp, 1] < 0) != (coords[mj, 1] < 0)]
        for dj in rng.choice(contra, size=min(c["mauthner->spinal"]["k"], len(contra)), replace=False):
            pre.append(mj); post.append(dj); w.append(float(c["mauthner->spinal"]["count"]))
    # spinal chain head -> tail: the bout travels down the cord
    sp_sorted = sp[np.argsort(coords[sp, 0])]
    for j in range(len(sp_sorted) - 1):
        for step in range(1, c["spinal->spinal"]["k"] + 1):
            if j + step < len(sp_sorted):
                pre.append(sp_sorted[j]); post.append(sp_sorted[j + step]); w.append(float(c["spinal->spinal"]["count"]))

    ids = np.arange(n, dtype=np.int64) + 200000
    groups = {k: [int(v) for v in vals] for k, vals in groups.items()}
    return (ids, coords, types.astype(str), np.array(pre, np.int32), np.array(post, np.int32),
            np.array(w, np.float32), groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="build the 60-neuron toy graph")
    ap.add_argument("--synthetic", nargs="?", const=7000, type=int, metavar="N",
                    help="build the fish-shaped synthetic brain (default 7000 neurons)")
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        ids, coords, types, pre, post, w, groups = smoke_graph()
    elif args.synthetic:
        ids, coords, types, pre, post, w, groups = synthetic_graph(args.synthetic)
    else:
        ids, coords, types, index = load_neurons()
        types_by_id = {rid: t for rid, t in zip(ids, types)}
        pre, post, w = load_edges(index, types_by_id)
        groups = load_groups(ids, index)

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
        json.dump({"source": source,
                   "label": {"smoke": "smoke graph", "synthetic": "synthetic graph"}.get(source, "Fish1 slice"),
                   "neurons": int(len(ids)), "synapses": int(len(pre)),
                   "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)

    print(f"neurons      {len(ids)}")
    print(f"edges (signed){len(pre)}")
    print(f"excitatory   {(w > 0).sum()}")
    print(f"inhibitory   {(w < 0).sum()}")
    print("groups:", ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"-> {BUILD/'graph.npz'} + {BUILD/'groups.json'}")


if __name__ == "__main__":
    main()