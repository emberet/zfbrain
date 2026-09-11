"""Devnet full dry-run — the same rig as sollive, but the air is FREE.

Every money rule is identical to sollive; the ONLY differences:

  * network: devnet, forced before the first RPC call (ZF_SOL_DEVNET=1,
    ZF_SOL_LIVE stripped) — this file cannot be pointed at mainnet
  * the keypair lives ONLY in RAM — a throwaway, never written to .env,
    never echoed; devnet air belongs to nobody
  * nothing may EVER broadcast: --send-sim only simulates deeper

python soldryrun.py             # sim-only, nothing broadcast (free)
python soldryrun.py --send-sim  # deeper devnet sim, still nothing broadcast
"""

import argparse
import json
import os
import sys

# devnet, before anything talks to an RPC
os.environ["ZF_SOL_DEVNET"] = "1"
os.environ.pop("ZF_SOL_LIVE", None)
os.environ.pop("SOL_RPC_URL", None)

from solders.keypair import Keypair  # noqa: E402

import sollive  # noqa: E402
import solrpc  # noqa: E402


def _airdrop_free(pub):
    """Devnet faucet: the chain gives air freely. Often rate-limited; that is
    an honest 'skipped', not a failure of the rig."""
    try:
        solrpc.airdrop(pub, 1_000_000_000)
    except Exception as e:  # noqa: BLE001
        print(f"airdrop   : skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send-sim", action="store_true",
                    help="deeper devnet sim, still nothing broadcast")
    args = ap.parse_args()

    assert solrpc.network() == "devnet", "dry-run must be on devnet"
    print(f"network   : {solrpc.network()}")

    kp = Keypair()  # RAM only
    pub = str(kp.pubkey())
    print(f"wallet    : {pub} (throwaway, RAM only)")

    bal = solrpc.get_balance(pub)
    print(f"balance   : {bal / 1e9:.4f} SOL")
    if bal < 1_000_000:
        print("# airdropping free devnet air...")
        _airdrop_free(pub)
        bal = solrpc.get_balance(pub)
        print(f"balance   : {bal / 1e9:.4f} SOL (after free air)")

    if not args.send_sim:
        print("# --send-sim to go deeper; nothing here ever broadcasts")
        return

    bh = solrpc.latest_blockhash()
    tx = sollive.launch_probe(kp, bh)
    err = solrpc.sim_err(sollive.simulate(tx))
    print(f"simulate  : {'OK (no error, nothing broadcast)' if not err else err}")
    os.makedirs("build", exist_ok=True)
    with open("build/dryrun.json", "w") as f:
        json.dump({"network": "devnet", "wallet": pub, "blockhash": bh,
                   "sim_err": err, "sent": False}, f, indent=2)
    print("# honest: simulateTransaction on devnet, spent 0, broadcast 0")
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()
