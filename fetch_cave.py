"""Pull the larval zebrafish whole-brain EM reconstruction out of CAVE into
data/raw/*.parquet — the data spike (NOTEPAD §1) and the export (§2).

Target: the Harvard/Google 7 dpf "connectomic resource" (Lichtman/Engert +
Google, bioRxiv 2025): 187,053 cell bodies, 41,175 molecularly typed neurons
(vglut2a / gad1b), 29.5 M axon→dendrite + 9.5 M axon→axon synapses, polarity
assigned to 21 M, registered to the Z-brain atlas. It lives in a CAVE
deployment; access is by request, and the datastack + table names have to be
read off the live server — that is what --spike does.

    export ZF_CAVE_DATASTACK=<name from the data-access email>
    export CAVE_AUTH_TOKEN=<your token>      # (or put both in .env)
    python fetch_cave.py --spike             # tables, counts, columns; writes nothing
    python fetch_cave.py --export TABLE      # TABLE -> data/raw/TABLE.parquet, chunked
    python fetch_cave.py --export TABLE --where "cell_type == 'Mauthner'"

Everything downstream (build_graph.py) reads parquet, never CAVE, so a
re-export is cheap and the graph build is reproducible from the files.
"""

import argparse
import os
import sys
from pathlib import Path

RAW = Path("data/raw")

# the paper's vocabulary; --spike prints what the server actually calls them
EXPECTED = {
    "cells": "one row per cell body: root id, soma xyz, Z-brain region, vglut2a/gad1b",
    "synapses": "pre root, post root, ad/aa type, polarity (exc/inh/other), position",
}


def _load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


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


def export(cc, table, where=None, chunk=500_000):
    """TABLE -> data/raw/TABLE.parquet in chunks (the synapse table is tens of
    millions of rows)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{table}.parquet"
    writer = None
    offset, total = 0, 0
    while True:
        kw = {"limit": chunk, "offset": offset}
        if where:
            kw["filter_equal_dict"] = None  # placeholder: caveclient filters are dict-based
        df = cc.materialize.query_table(table, **kw)
        if df is None or len(df) == 0:
            break
        if where:
            df = df.query(where)
        batch = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, batch.schema, compression="zstd")
        writer.write_table(batch)
        total += len(df)
        offset += chunk
        print(f"  {table}: {total:,} rows", file=sys.stderr)
        if len(df) < chunk:
            break
    if writer:
        writer.close()
    print(f"wrote {out} ({total:,} rows)")


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", action="store_true", help="census the datastack; writes nothing")
    ap.add_argument("--export", metavar="TABLE", help="dump TABLE to data/raw/TABLE.parquet")
    ap.add_argument("--where", help="pandas query applied to each chunk (e.g. \"polarity == 'exc'\")")
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()
    if not (args.spike or args.export):
        ap.print_help()
        return
    cc = client()
    if args.spike:
        spike(cc)
    if args.export:
        export(cc, args.export, args.where, args.chunk)


if __name__ == "__main__":
    main()
