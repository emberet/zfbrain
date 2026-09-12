"""A fake CAVE export, at Fish1's real scale, so the ingest path can be proved
without the token that does not exist yet.

`fetch_cave.py` has never run — CAVE refuses even the datastack list without
auth. Everything downstream of it (parquet -> build_graph -> memmapped graph ->
fishsim) therefore has no test at all, and the first time it would run for real
is on 39M rows of somebody else's data. This writes a table with the same
shape, the same column names, the same dtypes and the same awkward parts (a
polarity column that is null for 46% of rows, root ids that are not row
numbers, ids with gaps) and runs the whole path on it.

    python tools/fake_cave.py                 # small: 6k neurons, ~1.2M synapses
    python tools/fake_cave.py --full          # 187,053 neurons, ~39M synapses
    python tools/fake_cave.py --keep DIR      # leave the fixture on disk

What it checks:
  round-trip   with polarity on every row, the rebuilt graph is *exactly* the
               pair table it was generated from — same pairs, same weights
  polarity     with polarity on 54% of rows, measured wins where present,
               Dale's principle fills the rest, and the counts add up
  groups       all nine readout groups roam.py demands come back non-empty,
               and the four direction channels land on the same cells the
               synthetic graph puts them on
  csr          the memmapped graph's precomputed indptr matches what
               fishsim.to_csr would have computed by sorting
  simulate     FishSim loads it mmapped and runs frames without materialising
               the edge table
"""

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_graph as bg          # noqa: E402
import fishsim                     # noqa: E402

# what a synthetic population is called once it has been through a Z-brain
# registration. groups_from_regions has to find its way back from these.
REGION_OF = {
    "retina": "Retina - photoreceptor layer",
    "dsgc_up": "Tectum stratum periventriculare",
    "dsgc_down": "Tectum stratum periventriculare",
    "dsgc_left": "Tectum stratum periventriculare",
    "dsgc_right": "Tectum stratum periventriculare",
    "nmlf": "Nucleus of the medial longitudinal fasciculus",
    "vspn": "Vestibulospinal neurons",
    "mauthner": "Mauthner cell",
    "spinal": "Spinal cord",
    "other": "Hindbrain - rhombomere 4",
}
READOUT = ("retina", "dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right",
           "nmlf", "vspn", "mauthner", "spinal")


def _rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _strings(codes, values):
    """int8 codes (-1 = null) -> a *plain* arrow string column.

    The column has to be plain strings, because that is what a real CAVE export
    contains and the build's handling of it is the thing under test. But
    materialising 43M python strs to get there is what made the generator cost
    1.2 GB before it wrote a single row — the same heap fragmentation the build
    itself just had to be fixed for. Dictionary first, cast second: two strings
    per part instead of a million."""
    null = codes < 0
    idx = pa.array(np.where(null, 0, codes), pa.int8(), mask=null)
    return pa.DictionaryArray.from_arrays(idx, pa.array(values)).cast(pa.string())


def make_fixture(out, n_neurons=6_000, n_syn=1_200_000, polarity_frac=0.54,
                 seed=17, window=1_000_000):
    """Write data/raw-shaped parquet for a graph of n_neurons and about n_syn
    synapses. Returns the pair table it was generated from, so a build can be
    checked against it exactly.

    The wiring comes from build_graph.synthetic_graph rather than being random:
    a random graph would build fine and then simulate like noise, and the point
    is to exercise the readout groups and the pathways too. Each pair's |w| is
    its synapse count, and each synapse becomes one row — which is the shape a
    CAVE synapse table actually has, and the shape the CSV path wrongly assumed
    was already aggregated.
    """
    out = Path(out)
    rng = np.random.default_rng(seed)

    # pick the fan-in that lands on the requested row count: each pair expands
    # into |w| rows, and |w| averages about 4 across the rule table
# (helper defined above make_fixture; see _strings)
    probe_fanin = max(4, int(round(n_syn / max(1, n_neurons) / 4.0)))
    ids, coords, types, pre, post, w, groups = bg.synthetic_graph(
        n_neurons, seed=7, target_fanin=probe_fanin)
    counts = np.abs(w).astype(np.int64)
    counts[counts < 1] = 1
    total_rows = int(counts.sum())

    # --- neurons ---------------------------------------------------------
    region = np.empty(len(ids), dtype=object)
    for name, idx in groups.items():
        region[np.asarray(idx, dtype=np.int64)] = REGION_OF.get(name, "Forebrain")
    region[region == None] = "Forebrain"  # noqa: E711
    cell_type = np.where(types == "gaba", "gad1b", "vglut2a")
    cell_type[rng.random(len(ids)) < 0.05] = "unknown"   # the release types 41k of 187k
    ndir = out / "neurons"
    ndir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({
        "id": np.arange(len(ids), dtype=np.int64) * 3 + 11,   # ids with gaps, not row numbers
        "pt_root_id": ids.astype(np.int64),
        "pt_position_x": coords[:, 0].astype(np.float32),
        "pt_position_y": coords[:, 1].astype(np.float32),
        "pt_position_z": coords[:, 2].astype(np.float32),
        "cell_type": cell_type,
        "brain_region": region.astype(str),
    }), preserve_index=False), ndir / "part-000000000000.parquet", compression="zstd")

    # --- synapses --------------------------------------------------------
    sdir = out / "synapses"
    sdir.mkdir(parents=True, exist_ok=True)
    if sdir.exists():
        for stale in sdir.glob("*.parquet"):
            stale.unlink()
    sign_of_pair = np.sign(w).astype(np.int8)
    written, part, cursor = 0, 0, 0
    n_pairs = len(pre)
    while cursor < n_pairs:
        # take pairs until this window is about `window` synapse rows
        cum = np.cumsum(counts[cursor:cursor + window])
        take = int(np.searchsorted(cum, window) + 1)
        stop = min(n_pairs, cursor + take)
        c = counts[cursor:stop]
        pre_rows = np.repeat(ids[pre[cursor:stop]], c)
        post_rows = np.repeat(ids[post[cursor:stop]], c)
        sgn = np.repeat(sign_of_pair[cursor:stop], c)
        m = len(pre_rows)
        pol_codes = np.where(sgn > 0, 0, 1).astype(np.int8)
        if polarity_frac < 1.0:
            pol_codes[rng.random(m) >= polarity_frac] = -1
        pq.write_table(pa.table({
            "id": np.arange(written, written + m, dtype=np.int64) * 2 + 5,
            "pre_pt_root_id": pre_rows,
            "post_pt_root_id": post_rows,
            "ctr_pt_position_x": rng.integers(0, 60000, m).astype(np.int32),
            "ctr_pt_position_y": rng.integers(0, 30000, m).astype(np.int32),
            "ctr_pt_position_z": rng.integers(0, 20000, m).astype(np.int32),
            "size": rng.integers(50, 5000, m).astype(np.int32),
            "synapse_type": _strings((rng.random(m) >= 0.76).astype(np.int8), ["ad", "aa"]),
            "polarity": _strings(pol_codes, ["exc", "inh"]),
        }), sdir / f"part-{part:012d}.parquet", compression="zstd")
        written += m
        part += 1
        cursor = stop
    assert written == total_rows, (written, total_rows)

    # The pair table a correct build must reproduce, in the same index space.
    # synthetic_graph's rule table can emit the same (pre, post) twice — two
    # pathways that happen to pick the same cell — so the truth has to be
    # aggregated by pair the same way the build aggregates the rows.
    n = len(ids)
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    tot_syn = np.bincount(np.ravel(inv), weights=counts.astype(np.float64), minlength=len(uk))
    tot_w = np.bincount(np.ravel(inv), weights=w.astype(np.float64), minlength=len(uk))
    keep = (tot_syn >= bg.MIN_SYNAPSES) & (tot_w != 0)
    truth = {"pre": (uk[keep] // n).astype(np.int32), "post": (uk[keep] % n).astype(np.int32),
             "w": tot_w[keep].astype(np.float32)}
    return {"neurons": len(ids), "synapse_rows": written, "pairs": int(keep.sum()),
            "parts": part, "truth": truth, "groups": groups, "root_ids": ids,
            "coords": coords}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_roundtrip(fix, pre, post, w, stats):
    """With polarity on every row, the build must reproduce the pair table
    exactly. Not approximately — every pair, every signed weight."""
    t = fix["truth"]
    order_t = np.lexsort((t["post"], t["pre"]))
    order_g = np.lexsort((post, pre))
    assert len(pre) == len(t["pre"]), f"pairs: built {len(pre):,}, expected {len(t['pre']):,}"
    assert np.array_equal(pre[order_g], t["pre"][order_t]), "pre indices differ"
    assert np.array_equal(post[order_g], t["post"][order_t]), "post indices differ"
    assert np.allclose(w[order_g], t["w"][order_t]), "weights differ"
    assert stats["polarity_inferred"] == 0, stats
    assert stats["polarity_measured"] == stats["synapse_rows"], stats
    return stats


def check_polarity(fix, data_root):
    """With polarity on ~54% of rows the counts must still add up, and the
    measured half must not have been thrown away."""
    ids, coords, code, regions = bg.load_neurons(data_root)
    _, _, _, stats = bg.load_edges(data_root, ids, code)
    tot = (stats["polarity_measured"] + stats["polarity_inferred"]
           + stats["polarity_dropped"])
    assert tot == stats["synapse_rows"] - stats["dropped_unknown_endpoint"], (tot, stats)
    frac = stats["polarity_measured_frac"]
    assert 0.4 < frac < 0.7, f"measured fraction {frac} is not ~0.54"
    assert stats["polarity_inferred"] > 0, stats
    return stats


def check_groups(fix, data_root):
    ids, coords, code, regions = bg.load_neurons(data_root)
    groups, how = bg.load_groups(ids, coords, regions, data_root)
    missing = [g for g in READOUT if not len(groups.get(g, []))]
    assert not missing, f"roam.py would hard-exit: missing {missing}"
    assert "convention" in how, how
    # the four direction channels are a position split; on a fixture whose
    # regions came from the synthetic groups they must land on the same cells
    for g in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
        got = set(int(i) for i in groups[g])
        want = set(int(i) for i in fix["groups"][g])
        assert got == want, f"{g}: {len(got)} cells, expected {len(want)} (overlap {len(got & want)})"
    return {k: len(v) for k, v in groups.items()}, how


def check_csr_and_sim(build_root, frames=3):
    """The memmapped graph must load without materialising, its precomputed
    indptr must agree with the sort fishsim used to do, and it must simulate."""
    gpath = Path(build_root) / "graph.npz"     # the name; the dir sits next to it
    before = _rss_mb()
    g = fishsim.load_graph(gpath, Path(build_root) / "groups.json")
    assert isinstance(g["pre"], np.memmap), f"pre is {type(g['pre'])}, not a memmap"
    assert g.get("indptr") is not None, "no precomputed indptr — to_csr will re-sort"
    n = len(g["coords"])
    ref = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(np.asarray(g["pre"]), minlength=n), out=ref[1:])
    assert np.array_equal(np.asarray(g["indptr"]), ref), "indptr disagrees with the edge table"
    assert np.all(np.diff(np.asarray(g["pre"])) >= 0), "edges are not sorted by pre"
    load_cost = _rss_mb() - before

    sim = fishsim.FishSim(g)
    sim.set_drive(g["groups"]["retina"], 40.0)
    for gname in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
        sim.set_drive(g["groups"][gname], 12.0)
    t0 = time.time()
    for _ in range(frames):
        d = sim.run(400)
    return {"load_rss_mb": round(load_cost, 1),
            "s_per_frame": round((time.time() - t0) / frames, 3),
            "mean_hz": round(float(np.mean(list(d["rates_hz"].values()))), 1),
            "rates_hz": {k: round(v, 1) for k, v in d["rates_hz"].items()}}


# ---------------------------------------------------------------------------
def run(n_neurons, n_syn, keep=None):
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="zfb-fixture-"))
    root.mkdir(parents=True, exist_ok=True)
    data_a, data_b = root / "raw-full", root / "raw-partial"
    build = root / "build"
    ok = True
    try:
        print(f"# fixture at {root}")
        t0 = time.time()
        fix = make_fixture(data_a, n_neurons, n_syn, polarity_frac=1.0)
        print(f"  wrote {fix['synapse_rows']:,} synapse rows in {fix['parts']} parts "
              f"over {fix['neurons']:,} neurons ({time.time()-t0:.1f}s, "
              f"{sum(p.stat().st_size for p in data_a.rglob('*.parquet'))/1e6:.0f} MB)")

        t0 = time.time()
        _, _, _, pre, post, w, _, stats, _ = bg.build_real(data_a, build)
        gdir = build / bg.GRAPH_DIR
        size = sum(p.stat().st_size for p in gdir.glob("*.npy")) / 1e6
        print(f"  build        {len(pre):,} edges -> {size:.0f} MB in "
              f"{len(list(gdir.glob('*.npy')))} arrays "
              f"({time.time()-t0:.1f}s, peak RSS {_rss_mb():.0f} MB)")

        check_roundtrip(fix, pre, post, w, stats)
        print(f"  round-trip   OK — {len(fix['truth']['pre']):,} pairs reproduced exactly")
        del pre, post, w

        gcounts, how = check_groups(fix, data_a)
        print(f"  groups       OK — {gcounts}")
        print(f"               {how}")

        sim = check_csr_and_sim(build)
        print(f"  csr + sim    OK — load cost {sim['load_rss_mb']} MB RSS, "
              f"{sim['s_per_frame']}s/frame, mean {sim['mean_hz']} Hz")
        print(f"               {sim['rates_hz']}")
        print("               (rates are meaningless here: a fresh graph has no "
              "weight_scale,\n                so this is the EPSP/max|w| fallback. "
              "calibrate.py is what fixes that.)")

        fix2 = make_fixture(data_b, n_neurons, n_syn, polarity_frac=0.54, seed=18)
        p = check_polarity(fix2, data_b)
        print(f"  polarity     OK — {p['polarity_measured']:,} measured / "
              f"{p['polarity_inferred']:,} inferred / {p['polarity_dropped']:,} dropped "
              f"({p['polarity_measured_frac']:.1%} measured)")
        print(f"\n# peak RSS {_rss_mb():.0f} MB")
    except AssertionError as exc:
        ok = False
        print(f"\n!! FAILED: {exc}")
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"# kept at {root}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Fish1's real shape: 187,053 neurons, ~39M synapses (~1 GB, minutes)")
    ap.add_argument("--neurons", type=int, default=None)
    ap.add_argument("--synapses", type=int, default=None)
    ap.add_argument("--keep", default=None, help="write the fixture here and leave it")
    args = ap.parse_args()
    full = args.full or os.environ.get("ZFB_FIXTURE_FULL") == "1"
    n = args.neurons or (bg.REAL_NEURONS if full else 6_000)
    s = args.synapses or (38_974_036 if full else 1_200_000)
    raise SystemExit(0 if run(n, s, args.keep) else 1)


if __name__ == "__main__":
    main()
