# NOTEPAD

Running todo list for ZFBRAIN. Tracked here so it survives sessions. Items that
need a human (accounts, keys, servers, real money) are marked **[HW]**; an
agent can take everything else.

## 1 · Data spike — the gate for everything downstream **[HW FIRST]**
Fish1 lives in a CAVE database (still being proofread). Before the graph can
be real, confirm what is actually usable:

- [ ] Get a CAVE login + `CAVE_AUTH_TOKEN` (Fish1 browsing is public; refresh
      the auth token for a current materialization).
- [ ] Confirm the **datastack name** and live synapse/cell table names; update
      `fetch_cave.py` (currently assumes `fish1` + `live_query`).
- [ ] Confirm the synapse export columns (pre_root, post_root, coords, E/I =
      vglut2a / gad1b). Map them into `data/raw/neurons.csv`,
      `data/raw/synapses.csv`, `data/raw/groups.csv`.
- [ ] Pull the three circuits the release paper dissected: OMR (DSGC → ePT/lPT
      → nMLF → spinal CPG), Mauthner escape (M-cell + reticulospinal), hindbrain
      integrator.
- [ ] Decide graph size (5–20k neurons is the honest day-one target) and note
      the proofread percentage. If Fish1 is too thin, supplement with mapZebrain
      + Fish-X (see Research below).
- [ ] Verify usage terms: art project ≠ paper, so the Dec 2027 embargo on the
      early snapshot targets *publications* — but read `data_policy.html` and
      note what we can/ cannot publish on a live site.

## 2 · Build the graph  ***(build_graph.py is written; wire real data)***
- [ ] `python fetch_cave.py` works end to end.
- [ ] `python build_graph.py` on the real CSVs produces `build/graph.npz` +
      `build/groups.json`; printout sanity-checked (edge counts, E/I split).
- [ ] Wire the readout groups exactly: retina, dsgc_up/down/left/right, nmlf,
      vspn, mauthner, spinal. Missing groups = a hard error in roam.py.
- [ ] Fishbrain-style held-out check: a small "spiral test" proving
      percept → expected swim/escape before it ever points at the internet.

## 3 · Brain behavior
- [ ] Tune LIF constants / EPSP scale so the smoke test behaviors are sane
      (see `python fishsim.py`).
- [ ] Validate DSGC direction selectivity: injection on one side must steer
      that way, and the startle path must fire the Mauthner escape.
- [ ] Storyboard the 3 site behaviors on a webpage: optic-flow scroll, nMLF
      bout bursts, Mauthner dart-away. Tune thresholds.

## 4 · Internet creature
- [ ] Confirm headless Chromium + Playwright on your machine (the fly's stack).
- [ ] Allowlist edit: choose the link-rich domains the fish roams. Keep it
      small and public (Wikipedia family, Gutenberg, Open Library, arXiv,
      xkcd + the launchpad + blockscout + own coin page).
- [ ] Sanity-test the click veto against a fake "purchase" page.
- [ ] Decide hosting: single container like the fly; tunnel to the site
      (cloudflared quick tunnel; publish `site/web/live.json`).

## 5 · Memecoin half — the launch **[HW, money]**
- [ ] Verify pons launchpad selectors on the LIVE site (they drift; rhlive.py
      selectors are best-effort). Update `rhlive.py` accordingly.
- [ ] `python rhwallet.py new`, fund ~0.002 ETH on Robinhood Chain.
- [ ] `python rhdryrun.py` must print MATCHES + personal_sign ok, spending
      nothing.
- [ ] Choose ticket/name/description. Default scaffold is **ZFBRAIN /
      "Zebrafish Connectome"**. Set `ZF_RH_X` to a handle YOU own (the fly repo
      warns the same: using someone else's handle is impersonation).
- [ ] Dry-run once against the launchpad with a test token; keep it off the
      real name. Check the on-screen log labels "by fish / by rig".
- [ ] Live launch. Wait for the receipt **on camera** (rhlive does this).
      Read the receipt on blockscout yourself. Then set the contract address
      in `site/index.html` + live.json.
- [ ] Update the README "what is not real" for whatever the fish actually
      didn't do itself — that list is the whole point, don't soften it.
- [ ] Token bootstrapping is the creator's call. Decide pair (GOOGL default),
      tax (1%), supply 1,000,000,000 fixed at launch — mirror the fly.

## 6 · Voice
- [ ] Wire a real LLM key (`LLM_API_KEY`) and model for voice.py.
- [ ] Wire a posting endpoint (`POST_URL`) — X via a relay/tool of your choice.
      Posting is unsafe-empty by design: without POST_URL it just prints.
- [ ] Optionally: if you want the posts written in *your* voice (the @emberetme
      persona), hand that persona prompt to the draft system in voice.py.
- [ ] Watch for a session where a draft gets rejected — that means the number
      check is working; log one as proof in the repo.

## 7 · Site
- [ ] Deploy `site/` (Vercel / Netlify — pure static).
- [ ] Wire the live feed: either a published websocket from the roamer or the
      `site/web/live.json` file written by heartbeat (already implemented).
- [ ] Add the $ZFBRAIN contract + explorer link once launched.
- [ ] Keep the honest list green. Every "what is not real" claim must stay true.

## 8 · Nice-to-have (research backlog)
- [ ] Rheotaxis: whole-field reverse flow → swim against the current, as a
      gentle anti-founder-mode behavior.
- [ ] Whole-brain stretch: pull the full Fish1 synapse table once proofreading
      allows; the sim already loads any graph.
- [ ] Comparison angle nobody has: a fish and the fly doing the same page, side
      by side. Cheap if a fly node can run.
- [ ] Real reward circuit (the fish's analogue of the fly's mushroom body) —
      only after a real one exists; inventing one is out of scope.

## Key references (all read for this build)
- Fish1 release page + paper (release paper, bioRxiv 2025 / published 2026)
- simZFish — Science Robotics 2025 (OMR, lower-posterior field, rheotaxis)
- Fish-X — bioRxiv 2025 (neuromodulator-annotated zf reconstruction)
- mapZebrain / Svara Nature Methods 2022 (zf vEM, synapse detection)
- FlyWire whole-brain connectome + Shiu et al. 2024 (LIF conventions)
- ZAPBench / Chen 2018 (zebrafish functional recordings for validation)
- fruitflydev/flycoinrh + flybrain.online + FlyDegen (the patterns we copy and
  the honesty bar we match)

## Decisions recorded
- [x] Internet creature (not desktop / not torso-over-physics aquarium).
- [x] Circuit-first graph (behavior circuits now, whole brain later).
- [x] Fresh repo, mimic flycoinrh patterns rather than forking it.
- [x] Memecoin half included from day one, launch paced like the fly's:
      wallet → dryrun → test launch → the real one.