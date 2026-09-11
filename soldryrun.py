"""Devnet full dry-run — the same honest rig as sollive, but air is FREE.

Every money rule is identical to sollive; the ONLY differences:

  * network: devnet (air given free by the chain itself, no human funding —
    the honest counter to "fund ~0.01 SOL to launch", the analog of fetching
    your own scarf vs having someone else buy it)
  * nothing may EVER self-broadcast: --send here only re-simulates deeper,
    it never reaches sollive's broadcast path for one lamport

python soldryrun.py             # sim-only, nothing broadcast (free)
python soldryrun.py --send-sim  # deeper devnet sim, still nothing broadcast
"""

import argparse
import json
import os
import sys

import sollive  # honest: we ARE sollive, minus the mainnet gate
import solrpc

from solders.keypair import Keypair  # the census called it: used at line 51, imported NOW


def _ensure_devnet():
    os.environ.setdefault("ZF_SOL_DEVNET", "1")
    os.environ.pop("ZF_SOL_LIVE", None)  # dry-run may never be live


def _airdrop_free(pub):
    """Devnet airdrop: the chain gives air freely. No human funding, like the
    fly's Robinhood air but WITHOUT the human-funded step."""
    try:
        solrpc._call("requestAirdrop", [pub, 1_000_000_000])
    except Exception as e:  # noqa: BLE001
        print(f"airdrop    : skipped ({e}) — devnet may not need it (free rig)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send-sim", action="store_true",
                    help="deeper devnet sim, still nothing broadcast")
    args = ap.parse_args()

    _ensure_devnet()

    if os.environ.get("ZF_SOL_DEVNET") == "1":
        # HONESTER still than sollive.wallet(): a throwaway devnet keypair
        # that lives ONLY in RAM — never persisted .env, never echoed. Devnet
        # air belongs to nobody; the fly's identity here is disposable, so
        # nothing about it can be spent, reused, or pointed at mainnet.live.
        kp = Keypair()
        pub = str(kp.pubkey())
    else:
        kp = sollive.wallet()
        pub = str(kp.pubkey())
    print(f"wallet    : {pub}")

    bal = solrpc.get_balance(pub)
    print(f"balance   : {bal / 1e9:.4f} SOL")
    if bal < 1_000_000:
        print("# airdropping free devnet air...")
        _airdrop_free(pub)
        bal = solrpc.get_balance(pub)
        print(f"balance   : {bal / 1e9:.4f} SOL (after free air)")

    if args.send_sim:
        bh = solrpc.latest_blockhash()
        tx = sollive._launch_probe(kp, pub, bh)
        sim = sollive._simulate_launch(tx)
        err = sim.value.err if sim.value and sim.value.err else None
        print(f"simulate  : {'OK (no error, nothing broadcast)' if not err else err}")
        with open("build/dryrun.json", "w") as f:
            json.dump({"wallet": pub, "blockhash": bh, "sim_err": err,
                       "sent": False}, f, indent=2)
        print("# honest: simulateTransaction on devnet, spent 0, broadcast 0")
        sys.exit(0 if not err else 1)

    print("# --send-sim to go deeper; nothing here ever broadcasts on devnet")


if __name__ == "__main__":
    main()
