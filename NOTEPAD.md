# NOTEPAD

Running todo list for ZFBRAIN. Tracked here so it survives sessions. Items that
need a human (accounts, keys, servers, real money) are marked **[HW]**; an
agent can take everything else.

## 1 · Data spike — the gate for everything downstream **[HW FIRST]**
Three whole-brain larval EM reconstructions exist (all 2025 preprints, all
access-by-request). Decision (2026-09-11): build on the **Harvard/Google 7 dpf
"connectomic resource"** (Lichtman/Engert + Google): 187,053 cell bodies,
41,175 molecularly typed neurons (vglut2a/gad1b), 29.5 M axon→dendrite +
9.5 M axon→axon synapses, polarity for 21 M, Z-brain registered, on CAVE.
(Janelia Fish Fire&Wire, ~140k neurons + functional imaging, is the one with
the Dec-2027 paper embargo; Fish-X has the retina in the volume.)

- [ ] **[HW]** Request access to the Harvard/Google CAVE deployment; put
      `ZF_CAVE_DATASTACK` + `CAVE_AUTH_TOKEN` in `.env`.
- [ ] `python fetch_cave.py --spike` — census the datastack (tables, counts,
      columns); then `--export <cells>` and `--export <synapses>` to
      `data/raw/*.parquet` (written, untested until the token exists).
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
      `build/groups.json` + `build/graph.meta.json`; printout sanity-checked
      (edge counts, E/I split). The site header reads the meta file — the
      moment this runs, zfbrain.online stops saying "synthetic graph".
- [x] Until then: `build_graph.py --synthetic` — **187,053 neurons / ~39M
      synapses**, the larva's own counts, on the silhouette (same sampler as
      the site's hero), ~208 synapses per neuron, all nine populations wired
      and firing; stored as a 76-byte recipe and rebuilt in memory (3.4 s),
      weight scale from `calibrate.py` (2026-09-11).
- [ ] It runs hot: real pages drive it near the top of its stable band (~1/3
      of neurons lit, spinal 150-200 Hz, escapes more often than bouts). The
      fix is either the real connectome's structure or a homeostatic brake —
      shipped the honest number rather than invent a stabiliser.
- [ ] Wire the readout groups exactly: retina, dsgc_up/down/left/right, nmlf,
      vspn, mauthner, spinal. Missing groups = a hard error in roam.py.
- [ ] Fishbrain-style held-out check: a small "spiral test" proving
      percept → expected swim/escape before it ever points at the internet.

## 2b · Step 08 "Whole brain" — the audit, 2026-09-12
Everything below was checked against the repo today. Order matters: 1-3 are
yours, 4-11 are mine and none can be *tested* before 1 exists.

- [ ] **[HW] CAVE access.** `ZF_CAVE_DATASTACK` + `CAVE_AUTH_TOKEN` in `.env`.
      `fetch_cave.py` has never once run.
- [x] **Disk.** Was 240 MB free of 228 GB — worse than the 567 MB the
      pre-launch note claimed. Cleared 2026-09-12 to **37 GB** (regenerable
      caches + the 27 GB Adobe cache). The three `com.apple.os.update-*` APFS
      snapshots would not delete: they belong to software-update staging, not
      Time Machine, and clear when that update installs or is discarded.
      A 39M-synapse export is 1-2 GB of parquet before anything is built.
- [ ] **[HW] The embargo question**, still unanswered from §1. The Dec-2027
      embargo targets *publications*, but zfbrain.online is a live public site
      rendering the data. Read `data_policy.html` and decide what may be shown
      before the header stops saying "synthetic graph".
- [ ] **Format mismatch.** `fetch_cave.py` writes `data/raw/*.parquet`;
      `build_graph.py:41,56` reads `neurons.csv` / `synapses.csv`. A 39M-row
      synapse CSV is ~1.5 GB of text — the bridge has to be parquet-native.
- [ ] **`fetch_cave.py` `--where` is a stub**: it sets
      `filter_equal_dict = None` with a `# placeholder` comment, and the
      offset/limit pagination has never been checked against caveclient 8.2.1's
      real `query_table` signature.
- [ ] **The 39M-row index map.** `load_edges` does `df["pre"].map(index)` with
      a Python dict — minutes at this scale. `synthetic_graph`'s vectorised
      `connect()` is the pattern; `np.searchsorted` over sorted root ids.
- [ ] **Storage.** `np.savez_compressed` of 39M edges is the ~450 MB write that
      filled the disk and crash-looped the fish on 2026-09-11. Real data cannot
      use the 76-byte recipe trick: it needs memmapped `.npy` that `fishsim`
      opens without materialising, plus a `.gitignore` entry.
- [ ] **Readout groups.** `roam.py:_require_groups` hard-exits when any of
      retina / dsgc_up,down,left,right / nmlf / vspn / mauthner / spinal is
      empty. Z-brain registration gives regions, so most map straight across —
      **but dsgc_up/down/left/right is a real gap: direction selectivity is
      functional and EM does not contain it.** Either assign by arbor
      orientation as a stated proxy, or split the pretectal population and say
      on the site that the four directions are a convention, not a measurement.
      This is the one place the real connectome cannot simply replace the
      synthetic one, and it belongs in the honest list either way.
- [ ] **Polarity.** The release assigns polarity to 21M of 39M synapses. Sign
      the other 18M from the presynaptic cell's molecular type (vglut2a → +1,
      gad1b → −1) per Dale's principle — `load_edges` already has the
      mechanism. Publish the fraction inferred vs measured.
- [ ] **Recalibrate.** `calibrate.py` re-bisects `weight_scale` on the real
      graph. "It runs hot" may stop being true — real structure is the proposed
      fix for it — so re-check the site copy against the new numbers.
- [x] **The layout already works.** `roam.py:_graph_doc` projects a real EM
      volume in microns into the site's fish frame, and `graph_ver` busts the
      edge cache. `graph.meta.json` flips `label` from "synthetic graph" to
      "Fish1 slice" on its own; no site change needed.

## 3 · Brain behavior
- [x] Synaptic kernel fixed (one spike of weight w now delivers w mV, not
      ~4.5w); rates are per-neuron means; decode uses DSGC *asymmetry*, bouts
      and turns are rises above a running baseline, escape is a Mauthner burst
      (> ZF_ESCAPE_HZ) with corollary discharge after the fish's own bout.
      `python fishsim.py` asserts: all populations > 0 Hz, none saturated,
      still page = no bout/turn/escape, rightward drive steers right, a flash
      lifts Mauthner ≥ 5×, ten flashes habituate it (280 → 110 Hz).
- [x] Simulator at whole-brain scale: CSR + event-driven propagation in numba
      (`python fishsim.py --bench 180000`: 0.26 ms/step at 1 Hz, ~4 ms/step
      in runaway; a 400-step frame is 0.1–1.7 s at 180k neurons / 30 M
      synapses on the M4). Numpy fallback is the same maths, ~7× slower.
- [x] Habituation = short-term synaptic depression on the sensory neurons'
      outgoing synapses (`ZF_DEP_U` 0.002 per spike, `ZF_DEP_TAU` 20 s); the
      feed carries `brain.habituation`, the site shows it. No reward invented.
- [ ] Re-run the same checks on the Fish1 graph when it exists; the synthetic
      weights are hand-tuned (`SYN` in build_graph.py) and will not carry over.
- [ ] Storyboard the 3 site behaviors on a webpage: optic-flow scroll, nMLF
      bout bursts, Mauthner dart-away. Tune thresholds.

## 4 · Internet creature
- [x] Headless Chromium + Playwright confirmed; `roam.py` steps for real
      (screenshot → retina → 400 LIF steps → cursor), publishes `/state`,
      `/frame.jpg`, `/events` on 127.0.0.1:4660 (2026-09-11).
- [x] Allowlist widened 2026-09-12: **41 domains, 24 seeds** — reference and
      museums (Met, Public Domain Review, Library of Congress, Commons,
      Standard Ebooks, NASA, OSM), fish and brain science (ZFIN, mapzebrain,
      bioRxiv, eLife, EOL, microns-explorer), and the coin's venues. Crypto
      hosts are reachable but never seeds: Cloudflare + wallet prompts kill a
      life before its first hop. `archive.org` had been a seed that was never
      on the allowlist — every life starting there was fenced straight off it.
- [x] **Fence down (`ZF_ROAM_OPEN=1`, 2026-09-12, user's call.)** The floor
      under it applies either way and is not configurable off: private network
      (localhost, 127/8, 10/8, 192.168/16, 172.16-31, 169.254, `*.local`),
      `file:`/`chrome:`/`devtools:`/`view-source:`, and a deny-list
      (`ZF_BLOCKLIST`) for what must not appear on a public feed. Checked on
      the link's destination *before* the click — a page merely navigated away
      from has already been screenshotted and published once. Domain matching
      is a suffix compare now; `d in host` was accepting `arxiv.org.evil.com`.
- [ ] Sanity-test the click veto against a fake "purchase" page.
- [x] Hosting decided: this Mac runs the fish under launchd
      (`online.zfbrain.roam`), a named Cloudflare tunnel
      (`online.zfbrain.tunnel`) publishes it as `live.zfbrain.online`.
- [x] Tunnel `zfbrain-live` (5491648f…) created in the dashboard, public
      hostname `live.zfbrain.online → http://127.0.0.1:4660`, token in
      `.env` as `ZF_TUNNEL_TOKEN`, `online.zfbrain.tunnel` agent loaded — the
      feed is public (2026-09-11 19:46).
- [ ] **[HW]** Keep the Mac awake (`sudo pmset -c sleep 0` or Energy Saver).

## 5 · Memecoin half — the launch **[HW, money]**
- [x] `python solkeygen.py --env` → wallet in `.env`; funded 0.05 SOL.
- [x] `python soldryrun.py --send-sim` runs on devnet (faucet is flaky; a 0
      balance makes the sim say AccountNotFound — that is the honest result).
- [x] `python sollive.py --sim` signs a real 0-lamport self-transfer and
      simulates it on mainnet: OK, nothing broadcast; `--send` refuses
      without `ZF_SOL_LIVE=1`.
- [ ] The real create: `sollive.py` still signs only a probe — it never gained
      a pump.fun create instruction, and the launch below did not go through
      it. If the fish is ever to launch anything itself, this is still the
      missing piece.
- [x] Name/ticker chosen: **"Zebra Fish" / ZFBRAIN**, handle @zfbraindev.
- [x] **Launched 2026-09-12, by hand on pump.fun — not by the fish.**
      Mint `9eciHjJopku15zkke5GGdpPdfsDTqsfhQA9EibrApump`, Token-2022,
      6 decimals, supply 1,000,000,000, mint + freeze + update authority
      revoked, extensions `metadataPointer` + `tokenMetadata` only, so
      **no transfer fee** (the planned 1% / `ZF_SOL_TAX` never happened).
      Metadata on IPFS. Verify before quoting any of this:
      `python -c "import solrpc,json;print(json.dumps(solrpc._call('getAccountInfo',['9eciHjJopku15zkke5GGdpPdfsDTqsfhQA9EibrApump',{'encoding':'jsonParsed'}]),indent=2))"`
- [x] Site filled: `MINT` in `site/index.html` feeds the contract row, the copy
      button, the pump.fun / Solscan / Explorer links, step 07 and the
      phase label; the band reads SOL, bag, share and supply out of
      `/state`'s chain block, which `roam.py` refreshes every 60 s.
- [x] README's "what is not real" now says plainly that a person launched it.
      That list is the whole point — don't soften it.
- [ ] Whether the fish's bag is ever spent, and on what, is the creator's call
      and nothing in this repo can do it: no module holds a swap or transfer
      instruction.

## 6 · Voice
- [ ] **[HW]** `ANTHROPIC_API_KEY` in `.env` (voice.py uses the Anthropic SDK;
      `ZF_VOICE_MODEL` defaults to claude-sonnet-5). Without it `--dry` prints
      "no credentials" and stops.
- [x] Handle chosen: **@zfbraindev** (x.com/zfbraindev), linked in the site
      footer and set as `ZF_X_HANDLE` (2026-09-12).
- [ ] Wire a posting endpoint (`POST_URL`) — X via a relay/tool of your choice.
      Posting is unsafe-empty by design: without POST_URL it just prints.
- [ ] Optionally: if you want the posts written in *your* voice (the @emberetme
      persona), hand that persona prompt to `SYSTEM` in voice.py.
- [ ] Watch for a session where a draft gets rejected — that means the number
      check is working; log one as proof in the repo.

## 7 · Site
- [x] Deployed: Cloudflare Pages project `zfbrain`, custom domains
      `zfbrain.online` + `www` (2026-09-11). `zfsite.py --deploy` republishes
      (gated by `ZF_SITE_LIVE=1`; wrangler runs with an empty env file so the
      seed can never ride along).
- [x] `site/` is plain source now (`tools/unbundle.py` unpacked the design
      canvas once). Live panel subscribes to `live.zfbrain.online/events`,
      shows the real frame, retina grid toggle, real events; header counts
      the running graph; asleep state when nothing answers.
- [x] Live panel is public: zfbrain.online subscribes to
      live.zfbrain.online/events and shows the fish's frame (2026-09-11).
- [x] Hero is WebGL points fed by `/graph.bin` (float32 xy + uint16 group per
      neuron) and the heartbeat's `firing_mask` — sized for 187k neurons;
      real volumes get projected to the lateral view in roam.py.
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
- [ ] Viewers beyond ~200 concurrent SSE connections get 429 and fall back to
      polling; if launch day is bigger than that, front the feed with a Worker.

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
- [x] Solana / pump.fun, not Robinhood Chain (2026-09).
- [x] Live feed = named Cloudflare tunnel from the fish's own machine, not a
      store-and-forward Worker: real-time, no write caps, honest "asleep" when
      the machine is off (2026-09-11).
