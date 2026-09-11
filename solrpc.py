"""Thin Solana RPC client.

SOL_RPC_URL (default: devnet if ZF_SOL_DEVNET=1 else mainnet-beta), plus
SOL_RPC_API_KEY appended as the `api-key` header when set.

Every call here is honest read-only except send_raw_transaction / simulate
which are only reached by soldryrun.py (devnet, free) or sollive.py
(ZF_SOL_LIVE=1). Fees are SOL in lamports; a Token-2022 transfer-fee is a
protocol-level % on each transfer, configured at mint time, the honest analog
of the fly's tax.
"""

import os

import requests

RPC = os.environ.get(
    "SOL_RPC_URL",
    "https://api.devnet.solana.com" if os.environ.get("ZF_SOL_DEVNET") == "1"
    else "https://api.mainnet-beta.solana.com",
)
API_KEY = os.environ.get("SOL_RPC_API_KEY")


def _call(method, params=None, timeout=20):
    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["api-key"] = API_KEY
    r = requests.post(RPC, headers=headers,
                      json={"jsonrpc": "2.0", "id": 1, "method": method,
                            "params": params or []}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"{method}: {payload['error']}")
    return payload["result"]


def latest_blockhash():
    return _call("getLatestBlockhash")["value"]["blockhash"]


def recent_whitelisted():
    return _call("getTokenLargestAccounts")

# -- read-only ledger facts -----------------------------------------------
def get_balance(pubkey_b58):
    val = _call("getBalance", [pubkey_b58, {"commitment": "confirmed"}])
    return val["value"]


def get_token_largest(pubkey_b58):
    return _call("getTokenLargestAccounts", [pubkey_b58,
                                             {"commitment": "confirmed"}])


def airdrop(pubkey_b58, lamports):
    """requestAirdrop — devnet/devnet-faucet ONLY free air. This is the one
    method that can ADD lamports to a wallet, and it is honest the way the
    fly's self-rescued hero was: it can only ever be reached by soldryrun
    (devnet) which REQUIRES ZF_SOL_DEVNET=1, or by the free faucet directly.
    There is no mainnet 'air'; mainnet SOL arrives by human funding the
    address (NOTEPAD step), never by this function."""
    return _call("requestAirdrop", [pubkey_b58, lamports])


def simulate(raw_b58, sigs=None):
    """simulateTransaction: shows the outcome without committing anything."""
    return _call("simulateTransaction",
                 [raw_b58, {"encoding": "base58",
                            "replaceRecentBlockhash": True}])


def send_raw(raw_b58):
    """Only ever reached by sollive.py after ZF_SOL_LIVE=1."""
    return _call("sendTransaction", [raw_b58, {"encoding": "base58",
                                               "skipPreflight": True}])
