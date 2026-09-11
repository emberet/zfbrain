# ZFBRAIN

Larval-5 **zebrafish connectome** — a whole-vertebrate-brain project whose
cells and synapses are the real ones from the first fish brain ever published
(Fish1, Harvard Lichtman/Engert + Google): 187k neurons, 30M+ synapses,
download-on-demand, CC-BY research release.

Every "creature" behaviour runs on those actual circuits — there is no fake
brain talking to a fake market. There is a fish, and there is art.

## What it is

- a **retinotopic retina** (luminance + optic-flow DSGC channels, biased the
  way a larval OMR actually is)
- **real larva circuits**: DSGCs steer, the nMLF bout gate paces, vSPN flips
  direction, the Mauthner cell fires an all-or-nothing escape
- a **browser it reads through** — allowed sites only, veto list, no wallet,
  no keyboard, no downloads
- a **memecoin on Solana** (`$ZFBRAIN`) — Token-2022, transfer-fee tax,
  launchpad: pump.fun, secondary: Raydium

## The honest bit (kept loud on purpose)

- The graph is **circuit-first**, not finished. The full fish connectome is
  still being proofread; day one runs the circuits the release already
  maps and widens as the proofreading ships. `NOTEPAD.md` says exactly how
  wide day-one is.
- The **words are a narrator**, not the fish's language. Every post is an LLM
  given real telemetry + real token numbers, then number-checked; a draft
  whose numbers aren't in that packet is thrown away.
- **Nothing on this repo is financial advice.** It is developmental
  neuroscience as an art object.

## Layout

```
fetch_cave.py      Fish1 connectome -> data/raw/*.csv (download on demand)
build_graph.py     data/raw -> build/graph.npz + groups.json (--smoke works)
fishsim.py         LIF brain sim; behaviour readouts + smoke tests
retina.py          luminance + optic-flow retina
roam.py            the browser (allowlist, veto, heartbeat -> site/web/live.json)
voice.py           the narrator (observe -> read -> draft -> number-check -> post)
xpost.py           capped, deduped posting (safe with no POST_URL set)
solkeygen.py       Solana wallet (Ed25519, base58; address printed, seed never)
solrpc.py          thin Solana RPC surface (blockhash/balance/simulate/send)
sollive.py         mainnet launch driver — ZF_SOL_LIVE=1 gate, pump.fun launch
soldryrun.py       honest devnet full-sim twin (free air, nothing broadcast)
site/index.html    the status panel (at /state after roam starts)
NOTEPAD.md         the running todo list
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

# brain without data (works now):
python build_graph.py --smoke
python fishsim.py

# show a dry-run launch rig, free air, nothing broadcast:
cp .env.example .env
python solkeygen.py --env     # prints only the address
python soldryrun.py           # devnet sim, nothing ever broadcast
```

Two flags gate everything that touches the outside, both off: `ZF_ALLOW_BROWSER`
and `ZF_SOL_LIVE`. `soldryrun` is devnet-only by construction; `sollive --send`
is the only broadcast path and refuses to run without `ZF_SOL_LIVE=1`.

## Licence

Model after Shiu et al. 2024 / simZFish (Liu et al. 2025); connectivity from
the Fish1 / Lichtman-Engert research release (CC-BY). Own trackpads, own
decisions.
