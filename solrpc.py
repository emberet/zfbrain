"""Thin Solana RPC client.

The URL is resolved PER CALL from the environment, never at import time:
  SOL_RPC_URL           explicit override
  ZF_SOL_DEVNET=1       -> https://api.devnet.solana.com
  otherwise             -> https://api.mainnet-beta.solana.com
so soldryrun.py can force devnet before its first request even though it
imports this module at the top of the file. SOL_RPC_API_KEY, when set, goes
out as the `api-key` header.

Every call here is read-only except send_raw / simulate, which are only
reached by soldryrun.py (devnet, free) or sollive.py (ZF_SOL_LIVE=1). Fees
are SOL in lamports; a Token-2022 transfer-fee is a protocol-level % on each
transfer, configured at mint time, the honest analog of the fly's tax.
"""

import os

import requests

MAINNET = "https://api.mainnet-beta.solana.com"
DEVNET = "https://api.devnet.solana.com"


def rpc_url():
    if os.environ.get("SOL_RPC_URL"):
        return os.environ["SOL_RPC_URL"]
    return DEVNET if os.environ.get("ZF_SOL_DEVNET") == "1" else MAINNET


def network():
    return "devnet" if rpc_url() == DEVNET else "mainnet"


def _call(method, params=None, timeout=20, url=None):
    headers = {"content-type": "application/json"}
    api_key = os.environ.get("SOL_RPC_API_KEY")
    if api_key:
        headers["api-key"] = api_key
    r = requests.post(url or rpc_url(), headers=headers,
                      json={"jsonrpc": "2.0", "id": 1, "method": method,
                            "params": params or []}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"{method}: {payload['error']}")
    return payload["result"]


def latest_blockhash():
    return _call("getLatestBlockhash")["value"]["blockhash"]


def get_slot(url=None):
    """Current slot — the one live chain number the site header shows."""
    return int(_call("getSlot", [{"commitment": "confirmed"}], timeout=8, url=url))


# -- read-only ledger facts -----------------------------------------------
def get_balance(pubkey_b58):
    val = _call("getBalance", [pubkey_b58, {"commitment": "confirmed"}])
    return val["value"]


def get_token_largest(pubkey_b58):
    return _call("getTokenLargestAccounts", [pubkey_b58,
                                             {"commitment": "confirmed"}])


def airdrop(pubkey_b58, lamports):
    """requestAirdrop — devnet faucet ONLY, free air. This is the one method
    that can ADD lamports to a wallet and it only ever works on devnet; on
    mainnet the RPC rejects it. Mainnet SOL arrives by a human funding the
    address (NOTEPAD step), never by this function."""
    return _call("requestAirdrop", [pubkey_b58, lamports])


def simulate(raw_b58):
    """simulateTransaction: shows the outcome without committing anything.
    Returns the raw JSON result: {"context": {...}, "value": {"err": ..., ...}}."""
    return _call("simulateTransaction",
                 [raw_b58, {"encoding": "base58",
                            "replaceRecentBlockhash": True}])


def sim_err(sim):
    """The error (or None) out of a simulate() result."""
    return (sim or {}).get("value", {}).get("err")


def send_raw(raw_b58):
    """Only ever reached by sollive.py after ZF_SOL_LIVE=1. Takes base58."""
    return _call("sendTransaction", [raw_b58, {"encoding": "base58",
                                               "skipPreflight": True}])
