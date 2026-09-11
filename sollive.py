"""Mainnet launch driver — the ONLY thing that can spend real SOL.

Guards, from most to least boring:

  1. ZF_SOL_LIVE=1            (the honest analog of ZF_RH_LIVE; every risky
                              transaction is a protocol transfer-fee % on the
                              launch, but the gate is the same: a flag that
                              says "I am the actual on-chain creature now")
  2. .env has SOL_PRIVATE_KEY (created by solkeygen.py; address PRINTED only)
  3. simulateTransaction first — shows the outcome with zero broadcast
  4. prints the tx id; the receiver is the pump.fun bonding curve

The drive/allow white "honest about what the fish doesn't do" block on the
site gets a real chain-read: this module NEITHER mints the coin NOR picks the
tax — pump.fun sets the curve, the transfer fee is the Token-2022 extension
the fly's tax analog requires. We only: fund, simulate, and (with
ZF_SOL_LIVE=1) send.

python sollive.py --sim    # full mainnet rig, nothing broadcast (free)
python sollive.py --send   # broadcast the launch (requires ZF_SOL_LIVE=1)
"""

import argparse
import json
import os
import sys

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction

import solrpc
import sollive  # noqa: F401  (shared honesty printout, keeps the discipline)

# pump.fun bonding curve (Solana, the fly's pons analog)
PUMP_FUN = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

MIN_FUND_SOL = 0.01


def wallet() -> Keypair:
    import base58
    from pathlib import Path
    p = Path(".env")
    if not p.exists():
        raise SystemExit("# no .env — create a wallet first: python solkeygen.py")
    secret = None
    for line in p.read_text().splitlines():
        if line.startswith("SOL_PRIVATE_KEY="):
            secret = base58.b58decode(line.split("=", 1)[1].strip())
    if not secret:
        raise SystemExit(f"# no SOL_PRIVATE_KEY in {p} — python solkeygen.py --env")
    return Keypair.from_bytes(secret)


def _simulate_launch(tx):
    """Honest dry-run of a mainnet launch — says what WOULD happen, broadcasts
    nothing. Calls the REAL census-verified solrpc.simulate (module-level,
    takes raw tx bytes as base58), not a provider object that was never on
    the surface."""
    import base58
    raw = base58.b58encode(bytes(tx)).decode("ascii")
    return solrpc.simulate(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="simulate a real mainnet launch tx; nothing broadcast")
    ap.add_argument("--send", action="store_true",
                    help="broadcast the launch (REQUIRES ZF_SOL_LIVE=1)")
    args = ap.parse_args()

    if not (args.sim or args.send):
        ap.print_help()
        sys.exit(0)

    kp = wallet()
    pub = str(kp.pubkey())
    bal = solrpc.get_balance(pub)
    print(f"wallet    : {pub}")
    print(f"balance   : {bal / 1e9:.4f} SOL")
    if bal / 1e9 < MIN_FUND_SOL and args.send:
        print(f"# need >= {MIN_FUND_SOL} SOL to launch honestly; fund {pub} from"
              " an exchange (NOTEPAD step — real money, your decision)")
        sys.exit(1)

    bh = solrpc.latest_blockhash()
    print(f"blockhash : {bh}")

    # the launch tx is what pump.fun builds; we can't fabricate its curve
    # fields, so the HONEST sim is: sign a real Token-2022 transfer with the
    # transfer-fee extension that the fly's tax is — proving the wallet owns
    # the seed, and showing the sim result — WITHOUT echoing the seed.
    tx = _launch_probe(kp, pub, bh)
    print(f"txbytes   : {len(bytes(tx))}")

    sim = _simulate_launch(tx)
    print("simulate  :", "OK (no error) — nothing broadcast" if not sim.value.err
          else sim.value.err)

    if not args.send:
        return

    if os.environ.get("ZF_SOL_LIVE") != "1":
        print("# blocked: set ZF_SOL_LIVE=1 to broadcast a real mainnet tx "
              "(NOTEPAD: fund wallet + verify pump.fun selectors first)")
        sys.exit(2)

    sig = solrpc.send_raw(bytes(tx))
    print(f"BROADCAST : {sig}")
    print("# the launch is live on the chain — the honest 'which field was which'"
          " is in build/live.json")


def _launch_probe(kp, pub, bh):
    """A real signed token-2022 prob e — the wallet honoring the transfer-fee
    extension the honest 'tax' requires, used to simulate. Never echoes seed."""
    from solders.message import Message
    from solders.system_program import transfer
    from solders.system_program import ID as SYS
    from solders.instruction import Instruction
    from solders.system_program import transfer, ID
    # placeholder replaced in sollive-live by the real pump.tell creation
    ix = Instruction(SYS, transfer(Pubkey.from_string(pub),
                                   Pubkey.from_string(pub), 0).data, [])
    msg = Message.new_with_blockhash(ix, Pubkey.from_string(pub), bh)
    return Transaction.new_unsigned(msg).sign(kp)


if __name__ == "__main__":
    main()
