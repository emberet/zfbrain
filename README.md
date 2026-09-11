# ZFBRAIN

A living **larval-5 zebrafish connectome** on the open internet, funded with its
own memecoin. It reads pages through a retinotopic retina, steers itself through
a real browser, and — when told to — launches **$ZFBRAIN** on Solana through
**pump.fun**.

## Quickstart (everything honest, nothing broadcast by default)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# wallet (Ed25519 / Solana; prints the address only, NEVER the seed)
python solkeygen.py

# devnet full dry-run — free air, nothing broadcast, RAM-only keypair
python soldryrun.py

# the real launch rig (mainnet) — requires ZF_SOL_LIVE=1, still sims first
python sollive.py --sim
```

Two flags gate everything and both are **off by default**:

| flag | what it does |
|---|---|
| `ZF_SOL_LIVE=1` | lets `sollive.py --send` broadcast a real mainnet transaction |
| `ZF_ALLOW_BROWSER=1` | lets the roam brain actually open the web browser |

There is no other path. `soldryrun.py` is devnet-only and structurally cannot
broadcast; `sollive.py` simulates first, always, and refuses `--send` without
the gate.

## The honest bit (stated up front)

- The **graph is circuit-first**, not the finished brain. The full larval-5
  zebrafish connectome is still being proofread by the research release; day
  one runs the circuits the release already mapped and widens as proofreading
  completes. `NOTEPAD.md` says exactly how wide day-one is.
- The **words are a narrator**, not the brain's language. Every post is
  written by an LLM given the brain's real telemetry and the live token
  numbers, then number-checked; a draft whose numbers aren't in that packet is
  thrown away.
- **There is no internal goal.** No reward circuit feeds back; the goal is set
  outside and read honestly.
- **The launch rig completes the form.** The brain fills fields by texture and
  lands clicks through its real circuits; paired asset, tax and handle come
  from config, and `live.json` labels which was which.
- **$ZFBRAIN is an art experiment, not an investment.** Read the chain (Solscan
  / Solana Explorer) rather than taking this page's word.

## The brain (all real, all locally run)

```
build_graph.py   smoke graph + groups.json (--smoke works today)
fishsim.py       LIF whole-graph simulator; behavior readouts + tests
retina.py        luminance + optic-flow retina
roam.py          the roaming browser (allowlist, veto, heartbeat -> live.json)
voice.py         the narrator (observe -> read -> draft -> number-check -> post)
```

## The memecoin half (final surface, pump.fun on Solana)

- **Token-2022** with a **transfer-fee extension** — the tax is a real chain
  protocol fee, in basis points, set at create.
- `solkeygen.py` creates the Solana wallet (Ed25519, base58; **seed never
  echoed, address printed only**).
- `sollive.py` is the **only** module that can broadcast to mainnet — gated by
  `ZF_SOL_LIVE=1`, always `simulateTransaction` first, never echoes the seed.
- `soldryrun.py` is the honest twin: devnet, free air, RAM-only keypair, and by
  construction **cannot** broadcast.
- launch targets **pump.fun** (bonding curve) → **Raydium** migration.

## What is NOT real, stated plainly

- Not a finished whole-brain; it is circuit-first.
- Not the platform's own project, not affiliated with Fish1's authors or any
  exchange.
- Not financial advice; not an investment.

## Credits

Connectome: Fish1 (Lichtman/Engert labs + Google, CC-BY research release),
larval zebrafish. Model approach after Liu et al. 2025 (simZFish-like, for the
zebrafish). No affiliation with any brokerage.
