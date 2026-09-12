"""Build the connectome graph the sim runs on: data/raw/* -> build/graph/.

Larval zebrafish whole-brain connectome (Fish1, CC-BY research release).
We keep the circuits the release paper already dissected, drop pairs with
fewer than 3 synapses, and sign each edge from the molecular type of the
presynaptic neuron (vglut2a = excitatory, gad1b = inhibitory) exactly the way
the fly project signs from predicted neurotransmitters — but only where the
release did not measure the polarity itself. It measured 21M of 39M; the rest
is inference, and graph.meta.json records which is which.

Inputs (written by fetch_cave.py, or by hand). Each may be a parquet dataset
directory, a single .parquet, or a .csv — whichever exists, in that order:
  data/raw/neurons     id,x,y,z,type[,region]   type in {glu,gaba,unknown}
  data/raw/synapses    pre,post[,count][,sign|polarity]
  data/raw/groups      group,id                 readout wiring (optional)
CAVE's own column names (pt_root_id, pre_pt_root_id, pt_position_x, ...) are
accepted as aliases, so a raw --export needs no rewriting first.

Output:
  build/graph/*.npy      pre, post, w, coords, types, root_ids, indptr
  build/graph/graph.manifest.json
  build/groups.json      {group: [neuron index, ...]}
The arrays are written sorted by `pre` with the CSR indptr alongside, so
fishsim opens them with mmap and never re-sorts 39M edges at startup.

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
CALIB = "calibration.json"


def calib_key(meta):
    """What makes two graphs the same graph for calibration purposes."""
    return ":".join(str(meta.get(k)) for k in ("source", "neurons", "synapses", "fanin"))


def read_calibration(build_root, meta):
    """The stored calibration for this exact graph, or None."""
    p = Path(build_root) / CALIB
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get(calib_key(meta))
    except (OSError, ValueError):
        return None


def write_calibration(build_root, meta):
    """Record a calibration under this graph's identity, keeping the others.

    Separate from graph.meta.json on purpose. Every build rewrites the meta,
    so the meta is the wrong place to keep something that took a bisection to
    find: `build_graph.py --smoke` — one line in a test run — replaces the
    live synthetic graph's meta with the 60-neuron toy's and the calibrated
    scale is gone, which is how it was lost on 2026-09-12 *and* again while
    testing step 08. Keyed by graph, so a smoke build cannot evict the
    synthetic graph's entry and switching back restores it."""
    p = Path(build_root) / CALIB
    store = {}
    if p.exists():
        try:
            store = json.loads(p.read_text())
        except (OSError, ValueError):
            store = {}
    entry = {k: meta[k] for k in ("weight_scale", "calibrated_mean_hz", "calibrated_at")
             if k in meta}
    if not entry:
        return None
    store[calib_key(meta)] = entry
    p.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    return entry


def trace(label):
    """RSS at a labelled point, when ZFB_TRACE=1.

    This exists because the 39M-row build got the live fish jetsam-killed on an
    8 GB machine, and two rounds of *reasoning* about where the memory went
    were both wrong. Measure the build before trusting it near the fish."""
    if os.environ.get("ZFB_TRACE") != "1":
        return
    try:
        import resource
        import sys
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = raw / (1 << 20) if sys.platform == "darwin" else raw / 1024
        print(f"    [rss] {mb:7.0f} MB peak  {label}", flush=True)
    except Exception:
        pass


# CAVE names on the left of each tuple, ours on the right. The export is taken
# as it comes off the server; nothing has to be rewritten by hand first.
ALIASES = {
    # pt_root_id before id: a CAVE cells table carries both, and it is
    # pt_root_id that the synapse table's pre/post refer to. `id` there is the
    # annotation's own row id and matches nothing.
    "id": ("pt_root_id", "root_id", "id", "cell_id", "soma_id"),
    "x": ("x", "pt_position_x", "position_x", "soma_x"),
    "y": ("y", "pt_position_y", "position_y", "soma_y"),
    "z": ("z", "pt_position_z", "position_z", "soma_z"),
    "type": ("type", "cell_type", "molecular_type", "neurotransmitter", "classification_system"),
    "region": ("region", "brain_region", "zbrain_region", "area", "structure"),
    "pre": ("pre", "pre_pt_root_id", "pre_root_id", "pre_id"),
    "post": ("post", "post_pt_root_id", "post_root_id", "post_id"),
    "count": ("count", "n_synapses", "num_syn", "weight"),
    "sign": ("sign", "polarity", "sign_pred"),
}
# What a `type` / `polarity` cell can say, and what it means. Prefixes catch the
# long forms (vglut2a, gad1b, excitatory); the exact set catches the one-letter
# and numeric codings without also matching "endothelial" or "interneuron".
GLU_PREFIX, GABA_PREFIX = ("glu", "vglut", "exc"), ("gaba", "gad", "inh")
GLU_EXACT = {"e", "+1", "1", "1.0"}
GABA_EXACT = {"i", "-1", "-1.0"}


def _pick(columns, want):
    """Our column name -> the one this file actually uses, or None."""
    lower = {str(c).lower(): c for c in columns}
    for cand in ALIASES.get(want, (want,)):
        if cand in lower:
            return lower[cand]
    return None


def read_source(root, name, columns=None):
    """data/raw/<name> as a pandas frame, whatever shape it arrived in:
    a parquet dataset directory, a single parquet file, or a csv.

    Directory first, because that is what fetch_cave.py --export writes and the
    only one of the three that can hold 39M rows without a 1.5 GB text file."""
    root = Path(root)
    d, p, c = root / name, root / f"{name}.parquet", root / f"{name}.csv"
    if d.is_dir() and any(d.glob("*.parquet")):
        import pyarrow.parquet as pq
        return pq.read_table(d, columns=columns).to_pandas()
    if p.exists():
        import pyarrow.parquet as pq
        return pq.read_table(p, columns=columns).to_pandas()
    if c.exists():
        return pd.read_csv(c, usecols=columns) if columns else pd.read_csv(c)
    return None


def source_schema(root, name):
    """Column names without reading any rows, so a caller can decide which
    columns it wants before the first batch. Returns None for csv."""
    root = Path(root)
    d, p = root / name, root / f"{name}.parquet"
    files = sorted(d.glob("*.parquet")) if d.is_dir() else ([p] if p.exists() else [])
    if not files:
        return None
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(files[0]).schema_arrow.names)


def _iter_source(root, name, batch_rows=4_000_000, columns=None):
    """read_source, but one batch at a time so 39M synapse rows never all sit in
    pandas at once. Yields frames.

    `columns` matters more than it looks: a CAVE synapse export carries
    positions, sizes and a `synapse_type` string, and converting 43M arrow
    strings into python objects costs more memory than the whole graph. The
    build needs four columns; asking for four is the difference between a
    ~700 MB build and a ~3 GB one."""
    root = Path(root)
    d, p, c = root / name, root / f"{name}.parquet", root / f"{name}.csv"
    target = d if (d.is_dir() and any(d.glob("*.parquet"))) else (p if p.exists() else None)
    if target is not None:
        import pyarrow.parquet as pq
        ds = pq.ParquetDataset(target)
        for frag in ds.fragments:
            for batch in frag.to_batches(batch_size=batch_rows, columns=columns):
                # strings_to_categorical: a string column otherwise arrives as
                # a million python str objects per batch, and 39M of those
                # fragment the heap permanently (see _type_code)
                yield batch.to_pandas(strings_to_categorical=True)
        return
    if c.exists():
        for chunk in pd.read_csv(c, chunksize=batch_rows, usecols=columns):
            yield chunk
        return
    raise FileNotFoundError(
        f"{root/name}[/ .parquet .csv] missing. Run fetch_cave.py --export first, or use --smoke.")


def source_rows(root, name):
    """Row count from the parquet footers, without reading a single row.

    It lets load_edges allocate the three edge arrays once at the right size
    instead of appending per-batch and then concatenating, which at 43M rows
    means holding two full copies (~940 MB) at the moment of the concatenate.
    Returns None for csv, where there is no cheap count."""
    root = Path(root)
    d, p = root / name, root / f"{name}.parquet"
    files = sorted(d.glob("*.parquet")) if d.is_dir() else ([p] if p.exists() else [])
    if not files:
        return None
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def _type_code(series):
    """type column -> int8: +1 excitatory, -1 inhibitory, 0 unknown.

    Categoricals are decoded through their categories, not their rows. A 39M-row
    synapse table has a `polarity` string column with about three distinct
    values in it; classifying it row-by-row means creating and freeing 39M
    python str objects, which fragments the heap so badly that RSS climbs by
    1.5 GB and never comes back down. Three lookups and a take do the same job
    at constant cost. This is why _iter_source asks arrow for categoricals."""
    s = pd.Series(series)
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat_code = _type_code(s.cat.categories.to_numpy())
        codes = s.cat.codes.to_numpy()
        out = np.zeros(len(s), dtype=np.int8)
        seen = codes >= 0
        out[seen] = cat_code[codes[seen]]
        return out
    s = s.astype(str).str.strip().str.lower()
    code = np.zeros(len(s), dtype=np.int8)
    code[s.str.startswith(GLU_PREFIX).to_numpy() | s.isin(GLU_EXACT).to_numpy()] = 1
    code[s.str.startswith(GABA_PREFIX).to_numpy() | s.isin(GABA_EXACT).to_numpy()] = -1
    return code


def remap(values, sorted_ids):
    """Root ids -> row indices, vectorised. -1 for ids not in the neuron table.

    The dict-and-.map() this replaces was three python-level passes over the
    edge table; at 39M rows that is minutes. searchsorted is one C pass."""
    v = np.asarray(values, dtype=np.int64)
    if len(sorted_ids) == 0:
        return np.full(len(v), -1, dtype=np.int32)
    pos = np.searchsorted(sorted_ids, v)
    np.clip(pos, 0, len(sorted_ids) - 1, out=pos)
    out = pos.astype(np.int32)
    out[sorted_ids[pos] != v] = -1
    return out


def resolve_sign(measured, pre_idx, type_code):
    """One sign per synapse: the measured polarity where the release has it,
    Dale's principle off the presynaptic cell's molecular type where it does
    not, and 0 (= drop this synapse) where neither is known.

    Fish1 measures polarity for 21M of 39M synapses. Signing all 39M from
    molecular type would throw away the 21M that were actually looked at;
    signing only the 21M would throw away 46% of the brain. So: both, and
    report the split rather than implying it was all measured."""
    inferred = type_code[pre_idx].astype(np.float32)
    if measured is None:
        sign = inferred.copy()
        have = np.zeros(len(pre_idx), dtype=bool)
    else:
        m = np.asarray(measured, dtype=np.float32)
        have = np.isfinite(m) & (m != 0)
        sign = np.where(have, m, inferred).astype(np.float32)
    counts = {"measured": int(have.sum()),
              "inferred": int((~have & (sign != 0)).sum()),
              "unknown": int((sign == 0).sum())}
    return np.sign(sign).astype(np.float32), counts


def aggregate_pairs(pre_i, post_i, sign, n, min_synapses=MIN_SYNAPSES,
                    block_rows=4_000_000):
    """One row per synapse -> one edge per connected pair.

    A CAVE synapse table is one row per synapse, not the pre-aggregated pair
    table the CSV path assumed. So MIN_SYNAPSES is an aggregation step, not a
    filter: count the synapses between each ordered pair, drop pairs below the
    threshold, and let the weight be the signed sum — which equals ±count when
    a pair's synapses agree, and partially cancels when they do not.

    Written as sort + reduceat rather than np.unique(return_inverse=True)
    because the inverse is a second int64 array as long as the input — 350 MB
    at 43M rows, on top of the copy np.unique already makes internally.

    And done one block of presynaptic ids at a time, because even the sort is
    too big: an int64 key over 42M rows plus its argsort permutation plus the
    permuted copy is ~1 GB of transient on top of the inputs. The machine this
    runs on has 8 GB and a live fish in it; a 3 GB build gets the fish
    jetsam-killed, which is the crash-loop this was meant to prevent. Blocks
    are cut on row counts, not on equal id spans, so an uneven presynaptic
    distribution cannot produce one huge block. `pre` is a complete key: every
    row for a given pre lands in exactly one block, so blocking changes
    nothing about the result — and since blocks ascend, the output comes out
    already sorted by pre."""
    m = len(pre_i)
    if m == 0:
        z32, zf = np.zeros(0, np.int32), np.zeros(0, np.float32)
        return z32, z32, zf, 0
    sign = np.asarray(sign, np.float32)

    cum = np.cumsum(np.bincount(pre_i, minlength=n))
    marks = np.arange(block_rows, m, block_rows, dtype=np.int64)
    bounds = np.unique(np.concatenate(
        ([0], np.searchsorted(cum, marks, side="left") + 1, [n]))).astype(np.int64)
    del cum, marks

    out_pre, out_post, out_w = [], [], []
    n_pairs = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if len(bounds) == 2:
            take = slice(None)                       # one block: no mask at all
        else:
            take = (pre_i >= lo) & (pre_i < hi)
        key = pre_i[take].astype(np.int64)
        key *= np.int64(n)
        key += post_i[take]              # in place: no third full-length array
        s = sign[take]
        k = len(key)
        if not k:
            continue
        order = np.argsort(key, kind="stable")
        key = key[order]
        s = s[order]
        del order
        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        counts = np.diff(np.concatenate((starts, [k])))
        wsum = np.add.reduceat(s, starts)
        uk = key[starts]
        del key, s, starts
        n_pairs += len(counts)
        keep = (counts >= min_synapses) & (wsum != 0)
        uk, wsum = uk[keep], wsum[keep]
        out_pre.append((uk // n).astype(np.int32))
        out_post.append((uk % n).astype(np.int32))
        out_w.append(wsum.astype(np.float32))
        trace(f"    block [{lo},{hi}) {k:,} rows")
        del uk, wsum, counts, keep
    if not out_pre:
        z32, zf = np.zeros(0, np.int32), np.zeros(0, np.float32)
        return z32, z32, zf, int(n_pairs)
    return (np.concatenate(out_pre), np.concatenate(out_post),
            np.concatenate(out_w), int(n_pairs))


def load_neurons(root=None):
    """ids sorted ascending (so remap can searchsorted them), coords, type
    codes, and the region column if the export carries one."""
    root = Path(root or DATA)
    df = read_source(root, "neurons")
    if df is None:
        raise FileNotFoundError(
            f"{root/'neurons'}[/ .parquet .csv] missing. Run fetch_cave.py --export, or use --smoke.")
    c_id, c_type = _pick(df.columns, "id"), _pick(df.columns, "type")
    cx, cy, cz = (_pick(df.columns, k) for k in ("x", "y", "z"))
    if c_id is None or None in (cx, cy, cz):
        raise SystemExit(f"neurons table needs id and x,y,z; it has {list(df.columns)}")
    df = df.dropna(subset=[c_id])
    ids = df[c_id].astype(np.int64).to_numpy()
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    coords = df[[cx, cy, cz]].to_numpy(np.float32)[order]
    code = (_type_code(df[c_type]) if c_type else np.zeros(len(df), np.int8))[order]
    c_reg = _pick(df.columns, "region")
    regions = df[c_reg].astype(str).to_numpy()[order] if c_reg else None
    return ids, coords, code, regions


def load_edges(root, sorted_ids, type_code, batch_rows=4_000_000):
    """The synapse table -> (pre, post, w) as neuron-row indices and signed
    weights, plus the stats graph.meta.json publishes."""
    root = Path(root or DATA)
    n = len(sorted_ids)
    cap = source_rows(root, "synapses")
    if cap is not None:
        out_pre = np.empty(cap, np.int32)
        out_post = np.empty(cap, np.int32)
        out_sign = np.empty(cap, np.float32)
        filled = 0
    else:
        pres, posts, signs = [], [], []
    rows = dropped_ends = batches = 0
    tally = {"measured": 0, "inferred": 0, "unknown": 0}
    # ask for the four columns the build uses, not the whole synapse table
    names = source_schema(root, "synapses")
    want = None
    if names:
        want = [c for c in (_pick(names, "pre"), _pick(names, "post"),
                            _pick(names, "sign"), _pick(names, "count")) if c]
    for df in _iter_source(root, "synapses", batch_rows, columns=want):
        c_pre, c_post = _pick(df.columns, "pre"), _pick(df.columns, "post")
        if c_pre is None or c_post is None:
            raise SystemExit(f"synapse table needs pre and post; it has {list(df.columns)}")
        rows += len(df)
        pre_i = remap(df[c_pre].to_numpy(), sorted_ids)
        post_i = remap(df[c_post].to_numpy(), sorted_ids)
        ok = (pre_i >= 0) & (post_i >= 0)
        dropped_ends += int((~ok).sum())
        pre_i, post_i = pre_i[ok], post_i[ok]

        c_sign = _pick(df.columns, "sign")
        measured = None
        if c_sign is not None:
            col = df[c_sign]
            if pd.api.types.is_numeric_dtype(col.dtype):
                measured = np.sign(col.to_numpy(dtype=np.float32, na_value=np.nan))[ok]
            else:
                # classify the whole column first: on a categorical that is a
                # lookup over its ~3 categories, where masking first would
                # force every row back into a python string
                measured = _type_code(col).astype(np.float32)[ok]
            del col
        c_count = _pick(df.columns, "count")
        sign, part = resolve_sign(measured, pre_i, type_code)
        for k in tally:
            tally[k] += part[k]
        keep = sign != 0
        pre_i, post_i, sign = pre_i[keep], post_i[keep], sign[keep]
        if c_count is not None:
            # a pre-aggregated pair table: expand the count into the weight
            # rather than pretending each row is one synapse
            sign = sign * df[c_count].to_numpy(np.float32)[ok][keep]
        if cap is not None:
            k = len(pre_i)
            out_pre[filled:filled + k] = pre_i
            out_post[filled:filled + k] = post_i
            out_sign[filled:filled + k] = sign
            filled += k
        else:
            pres.append(pre_i)
            posts.append(post_i)
            signs.append(sign)
        del df, pre_i, post_i, sign
        if batches % 10 == 0:
            trace(f"  batch {batches} ({rows:,} rows)")
        batches += 1
    if cap is not None:
        pre_i, post_i, sign = out_pre[:filled], out_post[:filled], out_sign[:filled]
        del out_pre, out_post, out_sign
    else:
        pre_i = np.concatenate(pres) if pres else np.zeros(0, np.int32)
        post_i = np.concatenate(posts) if posts else np.zeros(0, np.int32)
        sign = np.concatenate(signs) if signs else np.zeros(0, np.float32)
        del pres, posts, signs
    trace("  streamed")
    pre, post, w, n_pairs = aggregate_pairs(pre_i, post_i, sign, n)
    trace("  aggregated")
    del pre_i, post_i, sign
    total = max(1, tally["measured"] + tally["inferred"] + tally["unknown"])
    stats = {"synapse_rows": rows,
             "dropped_unknown_endpoint": dropped_ends,
             "pairs_before_threshold": n_pairs,
             "min_synapses": MIN_SYNAPSES,
             "polarity_measured": tally["measured"],
             "polarity_inferred": tally["inferred"],
             "polarity_dropped": tally["unknown"],
             "polarity_measured_frac": round(tally["measured"] / total, 4)}
    return pre, post, w, stats


def load_groups(sorted_ids, coords=None, regions=None, root=None):
    """Readout groups. An explicit groups table wins; otherwise they are derived
    from the Z-brain region column."""
    root = Path(root or DATA)
    df = read_source(root, "groups")
    if df is not None and len(df):
        c_g = _pick(df.columns, "group") or "group"
        c_i = _pick(df.columns, "id")
        idx = remap(df[c_i].to_numpy(), sorted_ids)
        groups = {}
        for name, i in zip(df[c_g].astype(str).to_numpy(), idx):
            if i >= 0:
                groups.setdefault(name, []).append(int(i))
        if groups:
            return groups, "groups table"
    if regions is not None and coords is not None:
        return groups_from_regions(regions, coords)
    return {}, "none"


# Z-brain region name -> readout population. Substring match, first hit wins,
# so "Retina - photoreceptor layer" and "retina" both land on the retina.
REGION_MAP = [
    (("retina", "photorecep", "eye"), "retina"),
    (("mauthner", "m-cell", "m cell"), "mauthner"),
    (("nmlf", "medial longitudinal", "nucleus of the mlf"), "nmlf"),
    (("vestibulospinal", "vspn", "vestibular"), "vspn"),
    (("spinal", "cord"), "spinal"),
    (("tect", "pretect"), "dsgc"),          # split into four below
]


def groups_from_regions(regions, coords):
    """Z-brain regions -> the nine readout groups roam.py requires.

    Five of them — retina, nMLF, vSPN, Mauthner, spinal — are anatomy, and an
    atlas-registered volume names them directly. The other four are not:
    `dsgc_up/down/left/right` are *direction-selective* channels, and
    direction selectivity is a functional property that an electron-microscope
    volume does not contain. There is no measurement in Fish1 that says which
    tectal cell prefers which direction.

    So the tectal/pretectal population is divided into four quadrants of the
    retinotopic map by soma position — the same (x, z) split the synthetic
    graph uses — and that is recorded as a convention, not a result. The site
    has to say so."""
    regions = np.asarray([str(r).lower() for r in regions])
    assigned = np.zeros(len(regions), dtype=bool)
    groups = {}
    for words, name in REGION_MAP:
        hit = np.zeros(len(regions), dtype=bool)
        for wd in words:
            hit |= np.char.find(regions, wd) >= 0
        hit &= ~assigned
        groups[name] = np.flatnonzero(hit)
        assigned |= hit
    groups["other"] = np.flatnonzero(~assigned)

    d = groups.pop("dsgc")
    if len(d):
        cx, cz = coords[d, 0].mean(), coords[d, 2].mean()
        up, dn = coords[d, 2] >= cz, coords[d, 2] < cz
        lf, rt = coords[d, 0] < cx, coords[d, 0] >= cx
        groups["dsgc_up"], groups["dsgc_down"] = d[up & lf], d[dn & lf]
        groups["dsgc_left"], groups["dsgc_right"] = d[up & rt], d[dn & rt]
    else:
        for k in ("dsgc_up", "dsgc_down", "dsgc_left", "dsgc_right"):
            groups[k] = np.zeros(0, dtype=np.int64)
    # a larva has exactly one Mauthner pair; keep the two most lateral somata
    m = groups.get("mauthner", np.zeros(0, np.int64))
    if len(m) > 2:
        groups["mauthner"] = m[np.argsort(coords[m, 1])][[0, -1]]
    return ({k: [int(i) for i in v] for k, v in groups.items()},
            "z-brain regions; dsgc_up/down/left/right split by soma position "
            "(a stated convention — EM does not measure direction selectivity)")


# ---------------------------------------------------------------------------
# storage: a memmappable graph directory, not one compressed blob
# ---------------------------------------------------------------------------
GRAPH_DIR = "graph"
MANIFEST = "graph.manifest.json"


def write_graph_dir(out, ids, coords, type_code, pre, post, w):
    """Write the graph as one .npy per array, sorted by `pre`, with the CSR
    indptr alongside.

    np.savez_compressed of 39M edges is a ~450 MB write — the one that filled
    the disk and crash-looped the fish on 2026-09-11 — and np.load then
    materialises every array at every process start. Plain .npy files can be
    opened with mmap_mode='r' instead.

    Sorting here rather than in fishsim.to_csr is the other half of it: to_csr
    did np.argsort(pre) and then fancy-indexed post and w, which pulls the whole
    graph into memory and undoes the mmap. Sorted once at build time, that
    becomes a slice."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    arrays = {"root_ids": ids.astype(np.int64), "coords": coords.astype(np.float32),
              "types": np.asarray(type_code, np.int8), "pre": pre.astype(np.int32),
              "post": post.astype(np.int32), "w": w.astype(np.float32), "indptr": indptr}
    for name, arr in arrays.items():
        mm = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                       dtype=arr.dtype, shape=arr.shape)
        mm[...] = arr
        mm.flush()
        del mm
    manifest = {"format": 1, "neurons": int(n), "edges": int(len(pre)),
                "sorted_by": "pre", "has_indptr": True,
                "w_absmax": float(np.abs(w).max()) if len(w) else 0.0,
                "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                           for k, v in arrays.items()}}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


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



# ---------------------------------------------------------------------------
# where the neurons actually are: a larval zebrafish brain in 3-D
#
# The old layout filled a cartoon fish outline with dots, which renders as
# clip art. A 5-7 dpf larva's brain has a shape worth drawing: two enormous
# eyes, a pair of optic tecta over the midbrain, a hindbrain running back
# along the midline, the Mauthner pair in rhombomere 4, and a spinal cord
# tapering into the tail. Frame: x rostral->caudal (0..1 of body length),
# y left-right (0 = midline), z ventral->dorsal.
# ---------------------------------------------------------------------------
ANATOMY = {
    # A 5-7 dpf larva is ~4 mm long and almost all of that is tail: the brain
    # sits in the first fifth. Frame below is fractions of body length.
    # name:        centre (x, y, z),        radii (x, y, z),        shell, paired
    "eye":        ((0.097, 0.043, 0.000), (0.034, 0.024, 0.030), 0.55, True),
    "tectum":     ((0.146, 0.030, 0.015), (0.038, 0.026, 0.021), 0.45, True),
    "nmlf":       ((0.172, 0.012, -0.006), (0.018, 0.009, 0.010), 0.0, True),
    "vspn":       ((0.203, 0.013, -0.009), (0.020, 0.010, 0.010), 0.0, True),
    "mauthner":   ((0.214, 0.012, -0.006), (0.003, 0.0015, 0.002), 0.0, True),
    # one continuous mass from behind the eyes to the cord — without it the
    # eyes and tecta read as four loose balls instead of one brain
    "hindbrain":  ((0.158, 0.000, 0.002), (0.080, 0.030, 0.019), 0.0, False),
}
CORD = dict(x0=0.235, x1=1.0, r0=0.013, r1=0.0025)   # the cord tapers down the tail


def _blob(rng, n, centre, radii, shell=0.0, pair=False):
    """n points in an ellipsoid. `shell` (0..1) pushes them toward the surface —
    the retina is a cup and the tectum a sheet, so their cells sit on a shell,
    not through the middle. `pair` mirrors the blob about the midline."""
    if n <= 0:
        return np.zeros((0, 3), np.float32)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # radius^(1/3) fills a ball uniformly; biasing the exponent hollows it out
    r = rng.random(n) ** (1.0 / 3.0)
    if shell > 0:
        r = 1.0 - (1.0 - r) * (1.0 - shell)
    p = v * r[:, None] * np.asarray(radii)
    p += np.asarray(centre)
    if pair:
        # mirror half of them across the MIDLINE (y = 0), not about the blob's
        # own centre — a larva has an eye on each side, not two on one
        side = rng.random(n) < 0.5
        p[side, 1] *= -1.0
    return p.astype(np.float32)


def _cord(rng, n, x0, x1, r0, r1):
    """A tapering tube down the tail: the spinal cord."""
    if n <= 0:
        return np.zeros((0, 3), np.float32)
    t = rng.random(n) ** 0.85            # a little denser at the front
    x = x0 + t * (x1 - x0)
    r = (r0 + t * (r1 - r0)) * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    return np.stack([x, r * np.cos(a), r * np.sin(a) * 0.8], axis=1).astype(np.float32)


def larva_anatomy(n, sizes, rng):
    """Positions for every neuron, by population, in the shape of a larva.
    Returns coords (n,3) and the index arrays per group."""
    coords = np.zeros((n, 3), np.float32)
    groups, i = {}, 0

    def take(name, count, pts):
        nonlocal i
        idx = np.arange(i, i + count, dtype=np.int64)
        coords[idx] = pts
        groups[name] = idx
        i += count

    take("retina", sizes["retina"], _blob(rng, sizes["retina"], *ANATOMY["eye"]))
    # the tectum is one sheet per side; the four direction channels are read off
    # it by position, the way a real retinotopic map is divided
    tect = _blob(rng, sizes["dsgc"], *ANATOMY["tectum"])
    take("dsgc", sizes["dsgc"], tect)
    take("nmlf", sizes["nmlf"], _blob(rng, sizes["nmlf"], *ANATOMY["nmlf"]))
    take("vspn", sizes["vspn"], _blob(rng, sizes["vspn"], *ANATOMY["vspn"]))
    take("mauthner", sizes["mauthner"], _blob(rng, sizes["mauthner"], *ANATOMY["mauthner"]))
    take("other", sizes["other"], _blob(rng, sizes["other"], *ANATOMY["hindbrain"]))
    take("spinal", n - i, _cord(rng, n - i, **CORD))
    return coords, groups


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

    coords, groups = larva_anatomy(n, sizes, rng)

    # retina: index order (dorso-ventral, rostro-caudal) within the eyes, so the
    # 12x18 luminance grid lands on it as a visual field, not at random
    r = groups["retina"]
    groups["retina"] = r[np.lexsort((coords[r, 0], coords[r, 2]))]
    # the four direction channels are quadrants of the tectal map
    d = groups["dsgc"]
    cx, cz = coords[d, 0].mean(), coords[d, 2].mean()
    up, dn = coords[d, 2] >= cz, coords[d, 2] < cz
    lf, rt = coords[d, 0] < cx, coords[d, 0] >= cx
    groups["dsgc_up"], groups["dsgc_down"] = d[up & lf], d[dn & lf]
    groups["dsgc_left"], groups["dsgc_right"] = d[up & rt], d[dn & rt]
    del groups["dsgc"]
    # the Mauthner pair: one cell per side of the midline
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


def build_real(data_root, build_root):
    """The whole real-data path in one call, so the fixture test can drive it
    against a temp directory instead of the live build/."""
    trace("start")
    ids, coords, code, regions = load_neurons(data_root)
    trace("neurons")
    pre, post, w, stats = load_edges(data_root, ids, code)
    trace("edges")
    groups, how = load_groups(ids, coords, regions, data_root)
    stats["groups_convention"] = how
    manifest = write_graph_dir(Path(build_root) / GRAPH_DIR, ids, coords, code, pre, post, w)
    trace("written")
    with open(Path(build_root) / "groups.json", "w") as f:
        json.dump(groups, f)
    return ids, coords, code, pre, post, w, groups, stats, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="build the 60-neuron toy graph")
    ap.add_argument("--synthetic", nargs="?", const=REAL_NEURONS, type=int, metavar="N",
                    help=f"build the fish-shaped synthetic brain (default {REAL_NEURONS:,}, the larva's own count)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="drop a carried-over calibrated weight_scale and go back to the first guess")
    ap.add_argument("--data", default=None, help=f"input root (default {DATA})")
    ap.add_argument("--build", default=None, help=f"output root (default {BUILD})")
    args = ap.parse_args()

    data_root = Path(args.data) if args.data else DATA
    build_root = Path(args.build) if args.build else BUILD
    build_root.mkdir(parents=True, exist_ok=True)
    recipe = build_root / "graph.recipe.json"
    if recipe.exists():
        recipe.unlink()
    stats = {}
    if args.smoke:
        ids, coords, types, pre, post, w, groups = smoke_graph()
    elif args.synthetic:
        ids, coords, types, pre, post, w, groups = synthetic_graph(args.synthetic)
    else:
        ids, coords, types, pre, post, w, groups, stats, _ = build_real(data_root, build_root)

    if args.synthetic:
        # the synthetic graph is a pure function of (n, seed): storing 39M edges
        # would be ~450 MB of disk to reproduce something that builds in seconds,
        # so write the recipe and let load_graph regenerate it in memory
        with open(recipe, "w") as f:
            json.dump({"kind": "synthetic", "n": int(args.synthetic), "seed": 7,
                       "target_fanin": TARGET_FANIN}, f, indent=2)
        for stale in (build_root / "graph.npz", build_root / "groups.json"):
            if stale.exists():
                stale.unlink()
    elif args.smoke:
        # 60 neurons: the old single-file form is fine and the unit test reads it
        np.savez_compressed(
            build_root / "graph.npz",
            root_ids=ids,
            coords=coords,
            types=np.array([t.encode() for t in types]),
            pre=pre,
            post=post,
            w=w,
        )
        with open(build_root / "groups.json", "w") as f:
            json.dump({k: v for k, v in groups.items()}, f)
    # (the real path already wrote build/graph/ + groups.json inside build_real)
    # what the live site's header says about this graph — never a guess
    meta_path = build_root / "graph.meta.json"
    prev = {}
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            prev = {}
    with open(meta_path, "w") as f:
        source = "smoke" if args.smoke else ("synthetic" if args.synthetic else "fish1")
        meta = {"source": source,
                "label": {"smoke": "smoke graph", "synthetic": "synthetic graph"}.get(source, "Fish1 slice"),
                "neurons": int(len(ids)), "synapses": int(len(pre)),
                "fanin": round(len(pre) / max(1, len(ids)), 1),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        # how much of the polarity was measured and how much is Dale's principle,
        # plus what dsgc_up/down/left/right actually means. The site quotes these
        # rather than implying all 39M synapses were looked at.
        meta.update(stats)
        if source == "synthetic":
            # first guess at the weight scale: the same charge per neuron as the
            # small graph had, spread over this graph's fan-in. calibrate.py
            # refines it until fishsim's assertions pass.
            meta["weight_scale"] = round((4.0 / 6.0) * (8.4 / max(1.0, meta["fanin"])), 6)
        # ...but a calibrated scale outranks the guess. Rebuilding an identical
        # graph used to silently reset it, and the guess is not a mild
        # mis-tuning: at fan-in 208 the formula gives 0.0269, which calibrate.py
        # documents as *below the silent threshold*. That reset ran on
        # 2026-09-12 and left every motor pool — nMLF, vSPN, Mauthner, spinal —
        # flat at 0 Hz while the retina kept firing, so the fish clicked but
        # never swam. build/ is gitignored, so the only copy was the one
        # overwritten. Carry it across when the graph is the same graph.
        same = all(prev.get(k) == meta.get(k) for k in ("source", "neurons", "synapses", "fanin"))
        # the store first: it survives a build of a *different* graph in the
        # same directory, which the previous meta does not
        stored = None if args.recalibrate else read_calibration(build_root, meta)
        carry = stored or (prev if same and prev.get("calibrated_mean_hz")
                           and not args.recalibrate else None)
        # the store only ever holds calibrations, so a scale in it is enough;
        # a *meta* needs calibrated_mean_hz to tell a calibration apart from
        # the first-guess formula, which every synthetic meta also carries
        if carry and (carry.get("weight_scale") if stored else carry.get("calibrated_mean_hz")):
            for k in ("weight_scale", "calibrated_mean_hz", "calibrated_at"):
                if k in carry:
                    meta[k] = carry[k]
            where = "calibration.json" if stored else "the previous meta"
            print(f"# kept calibrated weight_scale={meta['weight_scale']} "
                  f"({carry.get('calibrated_at')}, from {where}); --recalibrate to drop it")
            write_calibration(build_root, meta)      # migrate meta-only entries
        elif prev.get("calibrated_mean_hz") and not same:
            print(f"# graph changed — dropping the old calibration "
                  f"(was {prev.get('weight_scale')}); run: python calibrate.py")
        json.dump(meta, f, indent=2)

    print(f"neurons      {len(ids)}")
    print(f"edges (signed){len(pre)}")
    print(f"excitatory   {(w > 0).sum()}")
    print(f"inhibitory   {(w < 0).sum()}")
    print("groups:", ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    if stats:
        print(f"polarity     {stats['polarity_measured']:,} measured / "
              f"{stats['polarity_inferred']:,} inferred / "
              f"{stats['polarity_dropped']:,} dropped "
              f"({stats['polarity_measured_frac']:.1%} measured)")
        print(f"groups from  {stats['groups_convention']}")
    out = (recipe if args.synthetic
           else (build_root / "graph.npz" if args.smoke else build_root / GRAPH_DIR))
    print(f"-> {out} + {build_root/'graph.meta.json'}")


if __name__ == "__main__":
    main()