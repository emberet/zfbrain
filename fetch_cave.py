"""Pull the larval zebrafish whole-brain EM reconstruction out of CAVE into
data/raw/<table>/*.parquet — the data spike (NOTEPAD §1) and the export (§2).

Target: the Harvard/Google 7 dpf "connectomic resource" (Lichtman/Engert +
Google, bioRxiv 2025): 187,053 cell bodies, 41,175 molecularly typed neurons
(vglut2a / gad1b), 29.5 M axon→dendrite + 9.5 M axon→axon synapses, polarity
assigned to 21 M, registered to the Z-brain atlas. It lives in a CAVE
deployment; access is by request, and the datastack + table names have to be
read off the live server — that is what --spike does.

    export ZF_CAVE_DATASTACK=<name from the data-access email>
    export CAVE_AUTH_TOKEN=<your token>      # (or put both in .env)
    python fetch_cave.py --spike             # tables, counts, columns; writes nothing
    python fetch_cave.py --export TABLE      # TABLE -> data/raw/TABLE/part-*.parquet
    python fetch_cave.py --export TABLE --where "cell_type == 'Mauthner'"

Everything downstream (build_graph.py) reads parquet, never CAVE, so a
re-export is cheap and the graph build is reproducible from the files.

Why id windows and not offset/limit
-----------------------------------
caveclient's query_table does take `offset`, but its own docstring says the
offset "will only return top K results" and the server promises no ORDER BY.
Paging a 39 M-row table by offset can therefore skip rows and repeat rows, and
you would never know — the total would look plausible. A window on the primary
key is a *set*: `lo <= id < hi` is the same set whatever order the server
returns it in, so windows can be retried, resumed, and counted against a
server-side count. That is the difference between a 39 M-row export you can
trust and one you cannot.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

RAW = Path("data/raw")

# the paper's vocabulary; --spike prints what the server actually calls them
EXPECTED = {
    "cells": "one row per cell body: root id, soma xyz, Z-brain region, vglut2a/gad1b",
    "synapses": "pre root, post root, ad/aa type, polarity (exc/inh/other), position",
}

# `where` -> the caveclient kwarg that means it. Order matters: the two-character
# operators have to be tried before the one-character ones.
_OPS = [
    ("==", "filter_equal_dict"),
    ("!=", "filter_out_dict"),
    (">=", "filter_greater_equal_dict"),
    ("<=", "filter_less_equal_dict"),
    ("~=", "filter_regex_dict"),
    (">", "filter_greater_dict"),
    ("<", "filter_less_dict"),
]
_IN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+[\(\[](.*)[\)\]]\s*$", re.I)
_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


def _literal(tok):
    """'Mauthner' / "exc" / 42 / 1.5 / true / null -> a python value."""
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    low = tok.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    if any(ch in tok for ch in "[]()"):
        # almost certainly a bounding box or a nested list. filter_spatial_dict
        # is not reachable from this mini-language; say so instead of quietly
        # turning it into the string "[[1,2,3],[4,5,6]]".
        raise ValueError(f"cannot parse the value {tok!r}; spatial/nested filters "
                         f"are not supported by --where, pass filter_spatial_dict in code")
    # a bare word: treat it as a string, which is what the user meant
    return tok


def parse_filters(where):
    """Turn a small `where` language into caveclient's filter_*_dict kwargs, so
    the filter runs on the server instead of after 39 M rows crossed the wire.

        "polarity == 'exc'"                  -> filter_equal_dict
        "cell_type in ('Mauthner','nMLF')"   -> filter_in_dict
        "size >= 200"                        -> filter_greater_equal_dict
        "a == 1 and b != 2"                  -> both, merged

    Raises on anything it cannot parse. That is deliberate: a `where` that
    silently does nothing means pulling the whole table by accident.
    """
    if not where or not where.strip():
        return {}
    out = {}
    for clause in re.split(r"\s+and\s+", where.strip(), flags=re.I):
        clause = clause.strip()
        if not clause:
            continue
        m = _IN_RE.match(clause)
        if m:
            col, body = m.group(1), m.group(2)
            vals = [_literal(t) for t in body.split(",") if t.strip()]
            if not vals:
                raise ValueError(f"empty `in` list: {clause!r}")
            out.setdefault("filter_in_dict", {})[col] = vals
            continue
        for op, kw in _OPS:
            i = clause.find(op)
            if i <= 0:
                continue
            col = clause[:i].strip()
            if not _COL_RE.match(col):
                continue
            rhs = clause[i + len(op):].strip()
            if not rhs or rhs[0] in "=<>~!":
                raise ValueError(f"cannot parse {clause!r}: operator soup around {op!r}")
            val = _literal(rhs)
            if kw == "filter_out_dict":
                # "not one of these" takes a list, not a scalar
                out.setdefault(kw, {}).setdefault(col, [])
                out[kw][col].append(val)
            else:
                out.setdefault(kw, {})[col] = val
            break
        else:
            raise ValueError(
                f"cannot parse {clause!r}. Supported: col == v, col != v, "
                f"col > v, col >= v, col < v, col <= v, col ~= regex, "
                f"col in (a, b), joined by `and`.")
    return out


def client():
    try:
        import caveclient
    except ImportError:
        raise SystemExit("caveclient not installed: .venv/bin/pip install caveclient")
    datastack = os.environ.get("ZF_CAVE_DATASTACK")
    token = os.environ.get("CAVE_AUTH_TOKEN")
    if not datastack:
        raise SystemExit("# no ZF_CAVE_DATASTACK — the datastack name comes with the data-access email")
    if not token:
        raise SystemExit("# no CAVE_AUTH_TOKEN — never echoed, lives in .env")
    server = os.environ.get("ZF_CAVE_SERVER")  # only if the deployment is not the default global server
    kw = {"auth_token": token}
    if server:
        kw["server_address"] = server
    return caveclient.CAVEclient(datastack, **kw)


def spike(cc):
    """Read-only census of the datastack: nothing is written."""
    print(f"datastack : {cc.datastack_name}")
    try:
        print(f"version   : {cc.materialize.version}")
    except Exception as exc:  # noqa: BLE001
        print(f"version   : ? ({exc})")
    tables = cc.materialize.get_tables()
    print(f"tables    : {len(tables)}")
    for t in tables:
        try:
            n = cc.materialize.get_annotation_count(t)
        except Exception:  # noqa: BLE001
            n = "?"
        try:
            meta = cc.materialize.get_table_metadata(t)
            desc = (meta.get("description") or "")[:70]
        except Exception:  # noqa: BLE001
            desc = ""
        print(f"  {t:40s} {str(n):>12}  {desc}")
    print("\nlooking for:")
    for k, v in EXPECTED.items():
        print(f"  {k:10s} {v}")
    print("\nnext: python fetch_cave.py --export <cells table> ; --export <synapse table>")


# ---------------------------------------------------------------------------
# counting and windowing
# ---------------------------------------------------------------------------
def precount(cc, table, filters=None):
    """How many rows the export should end up with, asked of the server.

    Cheap, and it is the only way to know afterwards that nothing was dropped.
    Returns None if the deployment does not support get_counts."""
    try:
        res = cc.materialize.query_table(table, get_counts=True, **(filters or {}))
    except Exception as exc:  # noqa: BLE001
        print(f"# get_counts unavailable ({exc}); the export will not be checked against a count",
              file=sys.stderr)
        return None
    if isinstance(res, dict):
        for k in ("count", "counts", "n", "total"):
            if k in res:
                return int(res[k])
        # a single-entry dict of whatever the server called it
        if len(res) == 1:
            try:
                return int(next(iter(res.values())))
            except (TypeError, ValueError):
                return None
        return None
    try:
        return int(res)
    except (TypeError, ValueError):
        return None


def range_semantics(cc, table, key):
    """Decide which kwarg actually means "key >= v" on this deployment.

    caveclient's own docstrings describe filter_greater_dict as an "exclusive
    upper-bound" and filter_less_dict as an "exclusive lower-bound", which are
    the wrong way round for the names. Rather than trust either reading, probe
    it: ask for rows on one side of a real value and look at what comes back.
    Returns (lo_kwarg, hi_kwarg) meaning (key >= lo, key < hi).
    """
    probe = cc.materialize.query_table(table, select_columns=[key], limit=64)
    if probe is None or len(probe) == 0:
        raise SystemExit(f"# {table} looks empty — nothing to export")
    vals = sorted(int(v) for v in probe[key].tolist())
    pivot = vals[len(vals) // 2]
    ge, lt = "filter_greater_equal_dict", "filter_less_dict"
    got = cc.materialize.query_table(table, select_columns=[key], limit=32, **{ge: {key: pivot}})
    if got is not None and len(got):
        seen = [int(v) for v in got[key].tolist()]
        if all(v >= pivot for v in seen):
            return ge, lt
        if all(v <= pivot for v in seen):
            # the names are inverted on this deployment: swap both
            print(f"# note: {ge}/{lt} are inverted on this server — swapping", file=sys.stderr)
            return "filter_less_equal_dict", "filter_greater_dict"
    print(f"# note: could not probe range semantics on {table}; assuming the names are literal",
          file=sys.stderr)
    return ge, lt


def max_key(cc, table, key, lo_kw, filters=None):
    """Smallest bound B with no rows at key >= B, found by doubling then
    bisecting. Uses counts only, so it never pulls rows."""
    def n_at_or_above(v):
        kw = dict(filters or {})
        kw.setdefault(lo_kw, {})[key] = v
        c = precount(cc, table, kw)
        if c is None:  # no counts: fall back to asking for a single row
            got = cc.materialize.query_table(table, select_columns=[key], limit=1, **kw)
            return 0 if got is None or len(got) == 0 else 1
        return c

    hi = 1 << 20
    while n_at_or_above(hi) > 0:
        hi <<= 1
        if hi > (1 << 62):
            raise SystemExit(f"# {key} on {table} is larger than 2^62 — not an integer key?")
    lo = hi >> 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if n_at_or_above(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def id_windows(hi, window):
    """[0, hi) chopped into half-open windows. A set, not a page."""
    lo = 0
    while lo < hi:
        yield lo, min(lo + window, hi)
        lo += window


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export(cc, table, where=None, window=250_000, key="id", out_root=RAW,
           resume=True, min_window=1_000):
    """TABLE -> data/raw/TABLE/part-<lo>.parquet, one file per id window.

    A directory rather than one file because pq.ParquetWriter pins the schema
    to the first batch it is handed: an all-null column in a later chunk aborts
    the write, and finding that out at row 20,000,000 of 39,000,000 is a bad
    afternoon. One file per window is also what makes --export resumable.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    filters = parse_filters(where)
    out = Path(out_root) / table
    out.mkdir(parents=True, exist_ok=True)
    (out / "_export.json").write_text(json.dumps(
        {"table": table, "where": where, "key": key, "filters": filters}, indent=2, default=str))

    expect = precount(cc, table, filters)
    if expect is not None:
        print(f"# {table}: server says {expect:,} rows", file=sys.stderr)

    lo_kw, hi_kw = range_semantics(cc, table, key)
    top = max_key(cc, table, key, lo_kw, filters)
    print(f"# {table}: {key} < {top:,}; {math.ceil(top / window):,} windows of {window:,}",
          file=sys.stderr)

    server_limit = None
    try:
        server_limit = int(cc.materialize.server_config.get("LIMIT") or 0) or None
    except Exception:  # noqa: BLE001
        pass

    total, files = 0, 0
    pending = list(id_windows(top, window))
    while pending:
        lo, hi = pending.pop(0)
        part = out / f"part-{lo:012d}.parquet"
        if resume and part.exists() and part.stat().st_size > 0:
            try:
                total += pq.ParquetFile(part).metadata.num_rows
                files += 1
                continue
            except Exception:  # noqa: BLE001
                part.unlink()          # half-written from an interrupted run
        kw = dict(filters)
        kw.setdefault(lo_kw, {})[key] = lo
        kw.setdefault(hi_kw, {})[key] = hi
        df = cc.materialize.query_table(table, split_positions=True, **kw)
        n = 0 if df is None else len(df)
        # a window that comes back exactly at the server's cap was truncated,
        # not completed: split it rather than silently losing the tail
        if n and server_limit and n >= server_limit and (hi - lo) > min_window:
            mid = (lo + hi) // 2
            print(f"  {table}: window [{lo:,},{hi:,}) hit the server cap — splitting",
                  file=sys.stderr)
            pending[:0] = [(lo, mid), (mid, hi)]
            continue
        if n:
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                           part, compression="zstd")
            files += 1
        total += n
        if files % 20 == 0 or n == 0:
            print(f"  {table}: {total:,} rows, {files} parts", file=sys.stderr)

    print(f"wrote {out}/ ({total:,} rows in {files} parts)")
    if expect is not None and total != expect:
        raise SystemExit(
            f"# COUNT MISMATCH: exported {total:,}, server counted {expect:,}. "
            f"Do not build a graph on this. Re-run with a smaller --window.")
    return out


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", action="store_true", help="census the datastack; writes nothing")
    ap.add_argument("--export", metavar="TABLE", help="dump TABLE to data/raw/TABLE/*.parquet")
    ap.add_argument("--where", help="server-side filter, e.g. \"polarity == 'exc'\"")
    ap.add_argument("--key", default="id", help="integer primary key to window on (default id)")
    ap.add_argument("--window", type=int, default=250_000, help="rows per id window")
    ap.add_argument("--no-resume", action="store_true", help="re-fetch windows already on disk")
    ap.add_argument("--check-where", action="store_true",
                    help="parse --where, print the caveclient kwargs, and exit; no network")
    args = ap.parse_args()
    if args.check_where:
        try:
            print(json.dumps(parse_filters(args.where), indent=2, default=str))
        except ValueError as exc:
            raise SystemExit(f"# {exc}")
        return
    if not (args.spike or args.export):
        ap.print_help()
        return
    cc = client()
    if args.spike:
        spike(cc)
    if args.export:
        export(cc, args.export, args.where, window=args.window, key=args.key,
               resume=not args.no_resume)


if __name__ == "__main__":
    main()
