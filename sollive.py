"""Mainnet signing rig — the ONLY thing here that can spend real SOL.

What it actually builds is a **0-lamport transfer from the wallet to itself**
(`launch_probe`). That is the whole transaction. It proves the seed signs, the
blockhash is fresh and the RPC accepts the tx; it creates nothing.

It has never contained a pump.fun create instruction. `PUMP_FUN` is a constant
nothing references. So $ZFBRAIN — launched 2026-09-12, mint
9eciHjJopku15zkke5GGdpPdfsDTqsfhQA9EibrApump — was **not** created by this
module or by the fish: a person created it by hand on pump.fun. The site says
so in its own words; don't let this docstring drift back into implying
otherwise. Building the real create instruction is still open work (NOTEPAD §5).

Guards, from most to least boring:

  1. ZF_SOL_LIVE=1            without it --send is refused
  2. .env has SOL_PRIVATE_KEY (created by solkeygen.py; address PRINTED only)
  3. simulateTransaction first — shows the outcome with zero broadcast
  4. prints the tx id

python sollive.py --sim    # full mainnet rig, nothing broadcast (free)
python sollive.py --send   # broadcast the probe (requires ZF_SOL_LIVE=1)
"""

import argparse
import os
import sys
from pathlib import Path

import base58
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction

import solrpc

# pump.fun bonding curve program (Solana)
PUMP_FUN = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

MIN_FUND_SOL = 0.01


def wallet() -> Keypair:
    """The fish's wallet from .env (or the SOL_PRIVATE_KEY env var). The seed
    is decoded in memory and never printed."""
    secret_b58 = os.environ.get("SOL_PRIVATE_KEY")
    if not secret_b58:
        p = Path(".env")
        if not p.exists():
            raise SystemExit("# no .env — create a wallet first: python solkeygen.py --env")
        for line in p.read_text().splitlines():
            if line.startswith("SOL_PRIVATE_KEY="):
                secret_b58 = line.split("=", 1)[1].strip()
    if not secret_b58:
        raise SystemExit("# no SOL_PRIVATE_KEY in .env — python solkeygen.py --env")
    return Keypair.from_bytes(base58.b58decode(secret_b58))


def launch_probe(kp: Keypair, bh: str) -> Transaction:
    """A real, signed 0-lamport transfer from the wallet to itself. Proves
    this process holds the seed (custody rail) and gives simulateTransaction
    something real to run, spending nothing. The actual create instruction
    is pump.fun's; it is not fabricated here."""
    pk = kp.pubkey()
    ix = transfer(TransferParams(from_pubkey=pk, to_pubkey=pk, lamports=0))
    blockhash = Hash.from_string(bh)
    msg = Message.new_with_blockhash([ix], pk, blockhash)
    tx = Transaction.new_unsigned(msg)
    tx.sign([kp], blockhash)
    return tx


def simulate(tx: Transaction):
    """Honest dry-run: says what WOULD happen, broadcasts nothing."""
    return solrpc.simulate(base58.b58encode(bytes(tx)).decode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="simulate a real mainnet tx; nothing broadcast")
    ap.add_argument("--send", action="store_true",
                    help="broadcast the launch (REQUIRES ZF_SOL_LIVE=1)")
    args = ap.parse_args()

    if not (args.sim or args.send):
        ap.print_help()
        sys.exit(0)

    kp = wallet()
    pub = str(kp.pubkey())
    print(f"network   : {solrpc.network()}")
    bal = solrpc.get_balance(pub)
    print(f"wallet    : {pub}")
    print(f"balance   : {bal / 1e9:.4f} SOL")
    if bal / 1e9 < MIN_FUND_SOL and args.send:
        print(f"# need >= {MIN_FUND_SOL} SOL to launch honestly; fund {pub} from"
              " an exchange (NOTEPAD step — real money, your decision)")
        sys.exit(1)

    bh = solrpc.latest_blockhash()
    print(f"blockhash : {bh}")

    tx = launch_probe(kp, bh)
    print(f"txbytes   : {len(bytes(tx))}")

    err = solrpc.sim_err(simulate(tx))
    print("simulate  :", "OK (no error) — nothing broadcast" if not err else err)

    if not args.send:
        return
    if err:
        print("# blocked: simulation failed; nothing sent")
        sys.exit(1)
    if os.environ.get("ZF_SOL_LIVE") != "1":
        print("# blocked: set ZF_SOL_LIVE=1 to broadcast a real mainnet tx "
              "(NOTEPAD: fund wallet + verify pump.fun flow first)")
        sys.exit(2)

    sig = solrpc.send_raw(base58.b58encode(bytes(tx)).decode("ascii"))
    print(f"BROADCAST : {sig}")
    print("# live on chain — read it on Solscan / Solana Explorer, not here")


if __name__ == "__main__":
    main()
