# ZFBRAIN

A living **larval zebrafish connectome** on the open internet, funded with its
own memecoin, a vertebrate brain and a different nervous system.

- read pages through a **retinotopic retina** (luminance + optic-flow DSGC
  channels, lower-posterior biased the way a larva's OMR is)
- behaves with the neurons a real larva uses: **DSGCs steer**, the **nMLF bout
  gate** bursts the scroll, **vSPN** flips it, and the **Mauthner cell** fires
  an all-or-nothing escape

## Layout

```
fetch_cave.py      Fish1 -> data/raw/*.csv (DCV: confirm datastack/tables)
build_graph.py     data/raw -> build/graph.npz + groups.json (--smoke works today)
fishsim.py         LIF whole-graph sim; behavior readout panels; smoke test
retina.py          luminance + 4-channel optic-flow retina
roam.py            the fish on the open internet (browser, fence, veto, /state)
rhwallet.py        create the wallet (writes to .env, prints only the address)
rhprovider.py      Robinhood Chain RPC (chain id 4663)
rhdryrun.py        prove sign + recover + estimate; never broadcasts
rhlive.py          drive the pons launchpad to a real launch (ZF_RH_LIVE=1)
voice.py           the narrator (observe -> read -> draft -> number-check -> post)
xpost.py           capped, deduped posting (safe to run with no POST_URL)
site/index.html    flybrain.online-style live page (static, reads web/live.json)
NOTEPAD.md         the running todo list for whoever drives this
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

# brain without data (works now):
python build_graph.py --smoke
python fishsim.py            # smoke: rightward input steers, strong input startles

# real data (NOTEPAD.md step 1 first):
python fetch_cave.py
python build_graph.py

# the fish on the internet (nowhere without the brain built):
cp .env.example .env          # set ZF_ALLOW_BROWSER=1
python roam.py                # http://localhost:4660/state

# the memecoin half:
python rhwallet.py new        # fund ~0.002 ETH on Robinhood Chain
python rhdryrun.py            # signing path, spends nothing
python rhlive.py              # real launch (needs ZF_RH_LIVE=1 too)
```

Two flags gate everything dangerous, both off by default: **`ZF_ALLOW_BROWSER=1`**
opens a browser against real sites, **`ZF_RH_LIVE=1`** signs and broadcasts a
transaction. Nothing else arms it.

## The rails, and why

The same reasoning as the fly's rails. The roaming browser has **no wallet, no
keyboard, no downloads**. Every click is checked before it lands and vetoed if
it reads as a submit, a pay wall, a connect or a sign-in — the veto count is on
the site. It roams an **allowlist** of link-rich public sites and its own
coin's pages; `ZF_ROAM_OPEN=1` removes that fence and should not be left on for
an unattended public stream.

## What is NOT real, stated plainly

- **The graph is a circuit-first slice**, not the finished brain — see above.
- **The words are a narrator.** The fish has no language. Every post is written
  by an LLM handed real telemetry and the live token numbers, and a draft with
  a number not in that packet — or with trading language — is thrown away.
- **There is no internal goal.** No reward circuit feeds back; the mushroom-body
  equivalence the fly has is out of scope until a real one exists.
- **The launch rig completes the form.** The fish fills fields by texture and
  lands clicks through its real circuits; paired asset, tax and handle come
  from config, and `live.log` labels which was which.
- **$ZFBRAIN is an art experiment, not an investment.**

## Credits

Connectome: Fish1 (Lichtman/Engert labs, Harvard + Google Research), CC-BY
research release; Fish-X (bioRxiv 2025) and mapZebrain as supplements. Model
approach after Shiu et al. 2024 and simZFish (Liu et al. 2025). Not affiliated
with Fish1's authors, pons, or Robinhood.
