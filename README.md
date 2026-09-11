# ZFBRAIN

A living **larval-5 zebrafish connectome** on the open internet, funded with its
own memecoin. It reads pages through a retinotopic retina, steers itself through
a real browser, and — when told to — launches **$ZFBRAIN** on Solana through
**pump.fun**. The site, **zfbrain.online**, shows what the fish is looking at
right now.

## Quickstart (everything honest, nothing broadcast by default)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

python build_graph.py --synthetic # 7,000-neuron fish-shaped synthetic brain (the stand-in)
python fishsim.py                 # the brain, no browser: every population fires, startle check

# wallet (Ed25519 / Solana; prints the address only, NEVER the seed)
python solkeygen.py

# devnet full dry-run — free air, nothing broadcast, RAM-only keypair
python soldryrun.py --send-sim

# the real launch rig (mainnet) — simulates only; --send needs ZF_SOL_LIVE=1
python sollive.py --sim
```

Flags gate everything and all are **off by default** (copy `.env.example` to
`.env`):

| flag | what it does |
|---|---|
| `ZF_ALLOW_BROWSER=1` | lets `roam.py` open a headless Chromium against real sites |
| `ZF_SOL_LIVE=1` | lets `sollive.py --send` broadcast a real mainnet transaction |
| `ZF_SITE_LIVE=1` | lets `zfsite.py --deploy` publish `site/` to zfbrain.online |

There is no other path. `soldryrun.py` forces devnet before its first RPC call
and cannot reach the broadcast path; `sollive.py` simulates first, always, and
refuses `--send` without the gate.

## Live (what the site shows)

```
this machine                                Cloudflare                     browser
roam.py ── 127.0.0.1:4660 ◄── cloudflared ── live.zfbrain.online ◄── SSE /events, /frame.jpg
                                              Pages (site/)  ◄── zfbrain.online
```

`roam.py` publishes, in-process, the heartbeat it just produced:

| route | what |
|---|---|
| `GET /state` | the heartbeat: graph size, page, cursor, retina, brain rates, decode, stats, last 12 events, mainnet slot |
| `GET /frame.jpg?seq=N` | what the fish is looking at (640×400 JPEG, one per step) |
| `GET /events` | the heartbeat as a Server-Sent-Events stream (≤2 Hz per viewer) |
| `GET /healthz` | 200 while the process is up |

The page subscribes to `/events`, falls back to polling `/state`, and when
nothing answers it says **asleep**. It never replays or invents telemetry; the
header counts the graph that is actually running.

Running it for real:

```bash
bin/roam.sh      # sources .env (needs ZF_ALLOW_BROWSER=1) and runs the fish
bin/tunnel.sh    # sources .env (needs ZF_TUNNEL_TOKEN) and runs the named tunnel
```

Both are wrapped as LaunchAgents (`~/Library/LaunchAgents/online.zfbrain.*.plist`,
`KeepAlive`, logs in `.state/`), so they restart on crash and come back after a
login. The feed is only live while this machine runs the fish; keep it awake.

Publishing the page:

```bash
python zfsite.py --sim       # census: files, and whether the live feed answers
ZF_SITE_LIVE=1 python zfsite.py --deploy   # wrangler pages deploy, empty env file
```

`site/` is plain static source (`index.html`, `styles/`, `fonts/`, `vendor/`),
unpacked once from the design-canvas bundle by `tools/unbundle.py`. Edit it
directly.

## The honest bit (stated up front)

- The **neurons are synthetic until Fish1 lands.** The full larval-5
  zebrafish connectome is still being proofread by the research release. What
  runs today is `build_graph.py --synthetic`: 7,000 LIF neurons laid out in
  the shape of the larva — retina, four DSGC channels, nMLF, vSPN, the
  Mauthner pair, a hindbrain integrator pool and a spinal cord — ~50,000
  signed synapses, every population firing. Every dot on the site's fish is
  one of them. The header always names the graph that is running, and it will
  say "Fish1" the day `build_graph.py` runs on the real CSVs.
- The **feed is live only while the fish is running** on one machine. Off, the
  page says asleep.
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
build_graph.py   graph.npz + groups.json + graph.meta.json (--synthetic today, --smoke for tests)
fishsim.py       LIF whole-graph simulator; behavior readouts + tests
retina.py        luminance + optic-flow retina
roam.py          the roaming browser (allowlist, veto) + the live feed server
voice.py         the narrator (observe -> read -> draft -> number-check -> post)
```

## The memecoin half (pump.fun on Solana)

- **Token-2022** with a **transfer-fee extension** — the tax is a real chain
  protocol fee, in basis points, set at create.
- `solkeygen.py` creates the Solana wallet (Ed25519, base58; **seed never
  echoed, address printed only**).
- `sollive.py` is the **only** module that can broadcast to mainnet — gated by
  `ZF_SOL_LIVE=1`, always `simulateTransaction` first, never echoes the seed.
- `soldryrun.py` is the honest twin: devnet, free air, RAM-only keypair, and by
  construction **cannot** broadcast.
- `solrpc.py` resolves the RPC URL per call, so the devnet/mainnet choice is
  made before any request goes out.
- launch targets **pump.fun** (bonding curve) → **Raydium** migration.

## What is NOT real, stated plainly

- Not Fish1 yet: a 7,000-neuron synthetic brain shaped like the larva, wired
  by hand from the circuits the release paper dissected.
- Not live when this machine is off.
- Not the platform's own project, not affiliated with Fish1's authors or any
  exchange.
- Not financial advice; not an investment.

## Credits

Connectome: Fish1 (Lichtman/Engert labs + Google, CC-BY research release),
larval zebrafish. Model approach after Liu et al. 2025 (simZFish-like, for the
zebrafish). No affiliation with any brokerage.
