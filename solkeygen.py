"""Create a Solana wallet (Ed25519) and write its secret to .env.

Prints ONLY the public address — the seed never leaves this machine and is
never echoed, exactly the fly/rh discipline. To later move real money you
fund this address with SOL; OS-currency note: on Solana gas is SOL and fees
are paid in SOL — the honest analog of the fly's Robinhood-gas. Ed25519,
base58 secret, no secp256k1/EVM anywhere.

python solkeygen.py --print   # show address to fund
python solkeygen.py --env     # write SOL_PRIVATE_KEY into .env
"""

import argparse
import base58
import sys
from pathlib import Path

import solders.keypair

ENV = Path(".env")
KEY = "SOL_PRIVATE_KEY"


def make_keypair():
    return solders.keypair.Keypair()


def _write_env(key, value):
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    out, found = [], False
    for line in lines:
        if line.startswith(key + "="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV.write_text("\n".join(out) + "\n")


def _secret_b58(kp):
    return base58.b58encode(bytes(kp)).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="show address only")
    ap.add_argument("--env", action="store_true",
                    help="write key to .env (asks once, echo-free)")
    args = ap.parse_args()

    kp = make_keypair()
    pub = str(kp.pubkey())
    print(f"address: {pub}")

    if not args.env:
        print("# fund this address with SOL — on devnet airdrop is free via soldryrun")
        return

    ans = input(f"write {KEY} to {ENV.name}? (y/N) ").strip().lower()
    if ans != "y":
        print("aborted, nothing written")
        sys.exit(1)
    _write_env(KEY, _secret_b58(kp))
    print(f"wrote {KEY} to {ENV.name} (base58 Ed25519 seed) — never echo it")


if __name__ == "__main__":
    main()
