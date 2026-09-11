"""Pull Fish1 (larval zebrafish whole-brain connectome) rows into data/raw/.csv.

Fish1 release: Lichtman/Engert (Harvard) + Google, CC-BY-ish research release.
The connectome lives in a live CAVE database and is still being proofread.
We only pull the three behavior circuits the release paper already dissects:
  * OMR visuomotor chain   (DSGC -> pretectum ePT/lPT -> nMLF -> spinal CPG)
  * Mauthner escape        (M-cells + reticulospinal interneurons)
  * Hindbrain integrator   (recurrent excitation + contralateral inhibition)

This script is the DATA SPIKE follow-up (see NOTEPAD.md). It is written
against the documented CAVEclient API, but the exact datastack name and
table/file names have to be confirmed against the live Fish1 deployment
before it will run end to end. Everything downstream of it (build_graph.py)
reads the CSVs it writes, never CAVE itself, so re-exporting is cheap.

Usage:
    export ZF_CAVE_DATASTACK=fish1            # confirm the real value
    export CAVE_AUTH_TOKEN=...                # google oauth token (caveclient)
    python fetch_cave.py                      # writes data/raw/*.csv

Fallbacks if CAVE is not usable yet:
    * mapZebrain / Svara Nature Methods 2022   (vEM volume w/ synapse detection)
    * Fish-X (bioRxiv 2025) neuromodulator-annotated reconstruction
    * FishExplorer exports (zib.de)           (single neuron SWC, connectivity)
"""

import csv
import os
from pathlib import Path

RAW = Path("data/raw")

CIRCUIT_QUERIES = {
    "circuit_omr": "retina OR dsgc OR pretectum OR (nmlf and reticulospinal)",
    "circuit_escape": "mauthner OR M-cell OR (reticulospinal and contralateral)",
    "circuit_integrator": "hindbrain AND (integrator OR recurrent)",
}


def main() -> None:
    datastack = os.environ.get("ZF_CAVE_DATASTACK", "fish1")
    RAW.mkdir(parents=True, exist_ok=True)

    try:
        import caveclient
    except ImportError:
        raise SystemExit(
            "caveclient not installed. pip install -r requirements.txt\n"
            "Then run the NOTEPAD data-spike to confirm datastack/table names."
        )

    cc = caveclient.CAVEclient(datastack=datastack)
    tables = cc.annotation.get_tables()
    print("Fish1 tables:", list(tables))

    # Schema will be confirmed in the data spike. Expected cols (aligned with
    # the main_text and supp methods of the Fish1 paper):
    #   pre_root / post_root, pre_pt_position_*, post_pt_position_*,
    #   neurotransmitter / vglut2a / gad1b one-hot per synapse.
    for name, where in CIRCUIT_QUERIES.items():
        out = RAW / f"{name}.csv"
        print(f"querying {name}: {where}")
        try:
            df = cc.materialize.live_query(query=where)
            if df is None or df.empty:
                print(f"  empty for {name}, skipping")
                continue
            df.to_csv(out, index=False)
            print(f"  wrote {out} ({len(df)} rows)")
        except Exception as exc:  # noqa: BLE001 - surface, don't die on spike
            print(f"  query failed for {name}: {exc}")

    # Soma positions + molecular type for every node we care about.
    # E/I comes from vglut2a (glutamatergic, excitatory) / gad1b (GABA, inhibitory).
    print("done. columns to confirm in the spike: pre_root, post_root, coords, E/I.")


if __name__ == "__main__":
    main()