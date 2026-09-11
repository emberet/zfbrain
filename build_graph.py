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
  python build_graph.py          # real data
  python build_graph.py --smoke  # tiny synthetic graph for developing the sim
"""

import argparse
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="build a tiny synthetic graph")
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        ids, coords, types, pre, post, w, groups = smoke_graph()
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

    print(f"neurons      {len(ids)}")
    print(f"edges (signed){len(pre)}")
    print(f"excitatory   {(w > 0).sum()}")
    print(f"inhibitory   {(w < 0).sum()}")
    print("groups:", ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"-> {BUILD/'graph.npz'} + {BUILD/'groups.json'}")


if __name__ == "__main__":
    main()