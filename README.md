# ZFBRAIN

A living **larval-5 zebrafish connectome** on the open internet, funded with its
own memecoin. It reads pages through a retinotopic retina, steers itself through
a real browser, and reads its own token's numbers back off Solana. **$ZFBRAIN**
is live on **pump.fun** —
`9eciHjJopku15zkke5GGdpPdfsDTqsfhQA9EibrApump` — launched by hand, not by the
fish. The site, **zfbrain.online**, shows what the fish is looking at right now.

## Quickstart (everything honest, nothing broadcast by default)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

python build_graph.py --synthetic # 187,053 neurons / ~39M synapses, the larva's own counts
python calibrate.py               # find the weight scale that keeps it in band
python fishsim.py                 # the brain, no browser: rates, startle, habituation

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

- **The counts are the larva's; the wiring is not.** A real 7 dpf zebrafish
  has **187,053 cell bodies and ~39 M synapses**, and that is what runs:
  `build_graph.py --synthetic` lays that many LIF neurons out in the shape of
  the larva, ~208 synapses each — retina, four motion channels, nMLF, vSPN,
  the Mauthner pair (exactly two, as in a real fish), a hindbrain integrator
  pool and a spinal cord. The *graph* is synthetic. The Harvard/Google 7 dpf
  EM reconstruction (187,053 cells, 39 M synapses, 21 M with polarity) is the
  target; `fetch_cave.py` pulls it once access is granted, and the header will
  name it the day it is in.
- **It runs hot.** A randomly wired graph this size has a narrow band between
  silence and saturation — `calibrate.py` finds it (silent at a weight scale
  of 0.030, saturated at 0.055), but real pages drive it near the top: about a
  third of neurons fire in any 5 ms window and the fish startles more than it
  swims. Structure is what keeps a real brain in band, and we do not have the
  real structure yet.
- The **only learning is habituation**: short-term synaptic depression on the
  sensory inputs, the real larval kind. A held flash stops startling the
  Mauthner cell; a page stared at for a minute drives the brain less. No
  reward signal is invented.
- The **feed is live only while the fish is running** on one machine. Off, the
  page says asleep.
- The **words are a narrator**, not the brain's language. Every post is
  written by an LLM given the brain's real telemetry and the live token
  numbers, then number-checked; a draft whose numbers aren't in that packet is
  thrown away.
- **There is no internal goal.** No reward circuit feeds back; the goal is set
  outside and read honestly.
- **The fish did not launch the token. A person did.** `sollive.py` can sign
  and simulate a mainnet transaction; it has never been able to build a
  pump.fun create instruction — read it. $ZFBRAIN was created by hand on
  pump.fun. The mint is real and the site reads its numbers back off chain;
  the claim that the animal did it would not be.
- **$ZFBRAIN is an art experiment, not an investment.** Read the chain (Solscan
  / Solana Explorer) rather than taking this page's word.

## The brain (all real, all locally run)

```
build_graph.py   the graph (--synthetic today, --smoke for tests); a synthetic
                 one is a 76-byte recipe, rebuilt in memory in seconds
calibrate.py     bisects the one number EM cannot give you: synapses -> mV
fishsim.py       LIF whole-graph simulator; behavior readouts + tests
retina.py        luminance + optic-flow retina
roam.py          the roaming browser (allowlist, veto) + the live feed server
voice.py         the narrator (observe -> read -> draft -> number-check -> post)
```

## The memecoin half (pump.fun on Solana)

- **Live since 2026-09-12**: `9eciHjJopku15zkke5GGdpPdfsDTqsfhQA9EibrApump`, a **Token-2022** mint — name "Zebra
  Fish", ticker ZFBRAIN, 6 decimals, supply 1,000,000,000, mint / freeze /
  update authority all revoked. Its only extensions are `metadataPointer` and
  `tokenMetadata`: **there is no transfer fee**, so nothing takes a cut of a
  transfer. Earlier drafts of this repo planned a 1% transfer-fee extension
  (`ZF_SOL_TAX`); that is not what was minted, and the site says so.
- The fish's wallet (`FDWcKEJjLbYP8bMrZ4XaR5uwbS1VFkzbVtw3wcxJys4t`) holds
  part of the supply; `roam.py` reads the balance, the bag and the supply off
  the public RPC once a minute and the site shows those numbers, not stored
  ones.
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

- Not a real connectome: the neuron and synapse counts are the larva's, the
  graph is synthetic, and it runs hotter than a larva does.
- Not live when this machine is off.
- Not the platform's own project, not affiliated with Fish1's authors or any
  exchange.
- Not financial advice; not an investment.

## Credits

Connectome: Fish1 (Lichtman/Engert labs + Google, CC-BY research release),
larval zebrafish. Model approach after Liu et al. 2025 (simZFish-like, for the
zebrafish). No affiliation with any brokerage.
