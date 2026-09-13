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
yours, 4-11 are mine.

**Update, same day:** items 4-9 are now built and tested — not against CAVE,
which is still auth-walled, but against a **fake CAVE-shaped parquet fixture**
(`tools/fake_cave.py`) at the real 187k-neuron / 39M-synapse shape. So "none can
be tested before 1 exists" turned out to be wrong, and the day the token lands
the remaining job is: `fetch_cave.py --spike` → `--export` ×2 →
`build_graph.py` → `calibrate.py`. Nothing in this pass publishes anything or
changes the graph the live fish loads.

- [ ] **[HW] CAVE access.** `ZF_CAVE_DATASTACK` + `CAVE_AUTH_TOKEN` in `.env`.
      `fetch_cave.py` has never once run. Probed today from the venv:
      `CAVEclient().info.get_datastacks()` → `AuthException: You have not setup
      a token`. **Even the list of datastacks is behind auth.** The token itself
      is free and self-serve (`client.auth.get_new_token()` → Google sign-in at
      `global.daf-apis.com/auth`) and is *not* the same thing as permission on
      the dataset — but with a token, `--spike` tells us whether the zebrafish
      datastack is already visible to you, which right now is unknown.
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
- [x] **Format mismatch.** Fixed. `read_source(root, name)` resolves
      `{name}.parquet` → `{name}/` dataset dir → `{name}.csv`, so the real pull
      is parquet-native and the hand-made CSV case still works. `load_edges`
      streams `pq.ParquetFile.iter_batches` into preallocated arrays sized from
      the parquet footers, so 39M rows never all sit in pandas.
- [x] **`fetch_cave.py` `--where` is a stub.** Fixed. `parse_filters` maps
      `==`, `!=`, `in (...)`, `>`, `>=`, `<`, `<=`, `~=` onto caveclient's eight
      `filter_*_dict` kwargs and **raises** on anything unparseable — a `where`
      that silently does nothing means pulling 39M rows by accident. Checked
      against the installed 8.2.1 signature. Two gotchas found and handled:
      caveclient's docstrings for `filter_greater_dict` / `filter_less_dict`
      are **inverted** ("upper-bound"/"lower-bound" swapped), so
      `range_semantics()` probes which kwarg means `>= v` at runtime instead of
      trusting either reading; and the offset/limit paging was unsafe — the
      server promises no ORDER BY, so over 39M rows it can silently skip and
      duplicate. Replaced with deterministic half-open id windows `[lo, lo+W)`,
      which are a *set* and therefore resumable, retryable, and checkable
      against a server-side `get_counts=True` precount. A window returning
      exactly the server cap is halved and retried; a final total ≠ precount is
      a hard `SystemExit`, not a warning.
- [x] **The 39M-row index map.** Fixed. `remap(values, sorted_ids)` is
      `np.searchsorted` plus a `sorted_ids[pos] == values` validity mask, `-1`
      for absent. Also: `build_graph.py` assumed a pre-aggregated `count`
      column, but a real CAVE synapse export is **one row per synapse** — so
      `MIN_SYNAPSES` is an aggregation, not a filter. `aggregate_pairs` does the
      group-by with `argsort` + `np.add.reduceat` (not
      `np.unique(return_inverse=True)`, whose int64 inverse is a second
      full-length array, ~350 MB at 43M rows).
- [x] **Storage.** Fixed. `write_graph_dir` writes one
      `np.lib.format.open_memmap` `.npy` per array plus `graph.manifest.json`.
      Edges are sorted by `pre` **once at build time** and the CSR `indptr` is
      written alongside, so `fishsim.to_csr` is now a no-op instead of doing
      `np.argsort(pre)` + fancy-indexing at every process start (which
      materialised the arrays and defeated the mmap). `w_absmax` in the
      manifest kills the `w.astype(float64)` upcast that cost ~600 MB every
      start just to find a max. Measured: **load cost 0.0 MB RSS at 9.5M
      edges** — the mmap genuinely works.
- [x] **Readout groups.** `roam.py:_require_groups` hard-exits when any of
      retina / dsgc_up,down,left,right / nmlf / vspn / mauthner / spinal is
      empty. Z-brain registration gives regions, so most map straight across —
      **but dsgc_up/down/left/right is a real gap: direction selectivity is
      functional and EM does not contain it.** Resolved the honest way:
      `groups_from_regions` maps Z-brain regions onto the nine populations, and
      splits the tectal/pretectal population into four quadrants by soma
      position — the same `(x, z)` centroid split the synthetic graph already
      uses. `meta["groups_convention"]` records in the data itself that the four
      directions are **a stated convention, not a measurement**, so the site can
      say so rather than implying the connectome measured them. This is the one
      place the real connectome cannot simply replace the synthetic one, and it
      stays in the honest list either way.
- [x] **Polarity.** Done. `resolve_sign` merges the two sources instead of
      treating them as either/or: measured where the release has it, Dale's
      principle from the presynaptic molecular type (vglut2a → +1, gad1b → −1)
      where it does not, dropped where the presynaptic type is unknown. It
      returns counts, so `graph.meta.json` carries `polarity_measured` /
      `polarity_inferred` / `polarity_dropped` and the site can state the
      inferred fraction. At full fixture scale: 23,376,649 measured /
      18,910,758 inferred / 1,009,679 dropped (54.0% measured).
- [ ] **Recalibrate.** `calibrate.py` re-bisects `weight_scale` on the real
      graph. "It runs hot" may stop being true — real structure is the proposed
      fix for it — so re-check the site copy against the new numbers. Still
      open, and with it the older question: `calibrate.py` uses `DRIVE = 40.0`,
      but real pages drive the retina at 12–16 Hz, below the knee. **[HW]** —
      recalibrating changes the creature's behaviour on a live public site.
- [x] **The fixture** — `tools/fake_cave.py`, the reason any of the above could
      be marked done without CAVE. It writes CAVE-shaped parquet windows with
      the real column names and dtypes and then drives fixture → `build_graph`
      → `fishsim.load_graph` → `fishsim.smoke()` in a temp dir, never touching
      `build/`. Default 6k/1.2M for speed; `--full` does the real 187k/39M
      shape. It has already earned its keep twice: it caught `ALIASES["id"]`
      preferring CAVE's annotation row id over `pt_root_id` (which the synapse
      table's `pre`/`post` never match — every edge would have silently
      vanished), and it caught the 3 GB build described below.
- [x] **The calibration got lost a second time — and is now hard to lose.**
      `4f03186` stopped a *rebuild* from resetting `weight_scale`, by carrying
      the old value across when the graph is the same graph. It read that value
      out of `build/graph.meta.json`. But every build rewrites that file, so
      one `build_graph.py --smoke` — a line in a test run, which is exactly
      what happened here on 2026-09-13 — replaces the live synthetic graph's
      meta with the 60-neuron toy's, and the calibrated scale is gone again.
      Same failure, different door. Three changes: calibrations now live in
      `build/calibration.json`, **keyed by graph identity**, so a smoke build
      cannot evict the synthetic graph's entry; `calibrate.py` writes there as
      well as to the meta; and `.gitignore` un-ignores that one file (`build/*`
      rather than `build/`, or the negation cannot reach inside), so "build/ is
      gitignored, so the only copy was the one overwritten" stops being true.
      Restored to 0.050141 and re-verified by `fishsim.smoke()`: flash takes
      Mauthner 0 → 210 Hz, habituation 210 → 120, no population silent or
      saturated. Checked: a rebuild → smoke build → rebuild now keeps it, and
      `--recalibrate` still drops it.
- [x] **Memory, found by running it.** The first full-scale run peaked at
      **3,352 MB** and macOS jetsam killed the live fish mid-build
      (`last exit reason = OS_REASON_JETSAM`, ~171 MB free pages). Two causes,
      both now fixed: `_iter_source` was converting *every* synapse column,
      including 43M `synapse_type` strings, when the build needs four — it now
      pushes the column list down into `frag.to_batches`; and
      `aggregate_pairs` sorted all 42M rows at once, ~1 GB of int64 key +
      permutation + permuted copy — it now aggregates one block of presynaptic
      ids at a time, cut on row counts so an uneven distribution cannot make
      one huge block. `pre` is a complete key, so blocking cannot change the
      result (checked against the unblocked path on random input at four block
      sizes), and blocks ascend, so the output arrives already sorted by pre.
      Then a third cause, found only because `ZFB_TRACE=1` now prints RSS at
      each phase: the `polarity` column arrives as pandas `object`, so
      classifying it row-by-row created and freed 43M python strs and
      fragmented the heap by 1.5 GB that never came back. `_iter_source` now
      asks arrow for categoricals and `_type_code` decodes the ~3 categories
      instead of the 43M rows. The fixture generator had the identical bug
      writing the column, fixed the same way (dictionary → plain string, so the
      files stay realistic). Net: standalone `build_graph.py` on 43M rows is
      **1,375 MB and 8.4s**, down from 2,434 MB and 18s; the whole fixture
      driver is **1,913 MB / 37s**, down from 3,352 MB / 74s. Two of the three
      causes were things I had reasoned about wrongly before measuring.
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

## 7b · The talk channel — built, shipped, pulled the same day (2026-09-13)
Asked for "a section where the fish can talk to people", built as a section
where people can *show* it something: a visitor's message drawn to a canvas and
drifted across the retina as an optomotor stimulus, with the reply being the
measured reaction (DSGC/nMLF/vSPN/Mauthner, bout/turn/startle) rather than
speech. A chat box was never an option — `#honest` item 5 says the fish has no
language, and it doesn't.

It worked end to end on the live fish through the public tunnel. **Then it was
removed, same day, by request.** The site section, `talk.py`, `_greet()`,
`POST /say`, `GET /talk` and `Voice.reply()` are all gone from HEAD; `ZF_TALK`
is out of `.env` and the process no longer has an inbound path.

- The whole thing is one command back: `git revert <the revert commit>`, or read
  it at **050b7d1**. Nothing about it was wrong — it passed every check.
- If it ever returns, the open question at removal was **cost exposure**: the
  optional narrator layer fires one Anthropic call per greeting, driven by
  strangers rather than by us. Ceiling was ~4,300 one-sentence calls/day at full
  saturation of the 20 s global floor. Gate it behind its own flag or a daily
  cap before re-enabling, rather than relying on the rate limits alone.
- Standing lesson, unrelated to the feature: backing up `.env` produced
  `.env.bak.*`, which `.gitignore`'s bare `.env` line did **not** match — the
  Solana seed briefly sat in an untracked, stageable file. `.env.*` now covers
  it. That fix stays.

## 7c · Bot walls — the fish was fleeing Wikipedia (2026-09-13)
Noticed the fish stuck: 1 page in 75 s, event ring nothing but
`look: no real links in reach` → `wander` → `escape`. Two separate bugs, both
found by actually loading the pages rather than reading the code.

**1. The interstitial changed under us.** `_is_captcha_page` was written for
"Checking your browser before accessing…". The current Cloudflare/Akamai
managed challenge is titled **"Just a moment..."** and says none of the old
strings, so it read as an ordinary page that happened to have two links. The
fish burned 15 quiet turns finding nothing clickable, wandered, and — three of
the 24 `HOME_SEEDS` being walled (`nga.gov`, `si.edu`, `loc.gov`; `eol.org`
since) — often landed straight on another one. Added the measured strings, and
`_walled` + `_pick_seed()` so a host that walls us drops out of the seed pool
for the rest of the process (not persisted: a restart gives every site a fresh
hearing, and the pool falls back to the full list rather than ever emptying).

**2. The word matching was hitting articles** — pre-existing, and the first fix
made it permanent by blacklisting the host. `"captcha"` and `"challenge"` are
substrings of ordinary titles, so **en.wikipedia.org/wiki/Challenger_Deep**
(1,931 links, 127 KB) was a "captcha wall" and the fish fled it. The whole of
`en.wikipedia.org` got dropped from the seed pool on the first run of the fix,
which is how it surfaced at all.

The discriminator is **thinness, not vocabulary**: every real wall measured had
<= 2 links and <= 300 bytes of text; the content pages that tripped the words
had 212-1,931 links and 11-127 KB. `WALL_MAX_LINKS = 8` / `WALL_MAX_TEXT =
2500` sits orders of magnitude clear of both. Verified 4/4 walls caught, 20/20
real pages clean, including all four Wikipedia articles that used to trip.

- Throughput after: **16 pages in the first 75 s**, against 1 before.
- Worth re-running that probe occasionally. These strings are a snapshot of
  what the CDNs served on one day; the last set lasted until it didn't, and the
  failure mode is silent — the fish just gets quietly worse at browsing.
- [x] **Pruned, same day.** Probed all 24 seeds rather than only the four the
      fish had happened to hit: exactly those four were walled, no others, none
      errored. `HOME_SEEDS` is now 20. They stay on `DEFAULT_ALLOWLIST` on
      purpose — reaching them by *wandering* is still fine and the runtime
      darts away; the seed list only decides where a life may **begin**.
      Verified after: 27 pages in 300 s, zero walls hit, no deaths.
- The seed-list comment claimed "no captcha-walled sites here" and had been
  wrong for some time. Re-probe rather than trust it; the probe is
  `/tmp/zfb-seeds.py` in shape — 15 lines of Playwright over `HOME_SEEDS`.
- `archive.org` renders **0 `a[href]`** at domcontentloaded + 2.5 s (it hydrates
  late). Not a wall, so it survived the prune, but a life starting there has
  nothing to click until it hydrates. Left alone; worth a look if it shows up
  as a dead seed.

## 7d · Plasticity + the table (2026-09-13)

Two asks, one feature: give the fish more to learn with, and a place to play
table tennis with it. They belong together because a rally is the first thing
this project has ever had that **repeats** — the same stimulus, the same cells,
over and over — which is the only condition under which plasticity is
observable at all. Reward-free by decision: `fishsim.py`'s "No reward signal is
invented" and `#honest` item 5 are unchanged, verbatim.

### The measurements the design rests on (all verified, none assumed)

- **The plastic set is 40% of the brain, not a corner of it.** retina n=48,091,
  dsgc n=26,730; retina outgoing 6,816,150 edges, dsgc outgoing 7,782,350 →
  14,598,500 = **37.5%** of 38,974,036.
- **`groups["retina"]` is exactly `arange(0, 48091)`**, so its CSR rows are the
  single contiguous slice `weights[:indptr[48091]]`. That is what makes a
  pre-gated Hebbian rule possible with **no reverse index** — STDP would have
  needed an inverted copy of 14.6M edges, 60–120 MB. Rejected on that.
- **Every retina→DSGC weight starts at one magnitude, 0.30084598** (= 6 ×
  0.050141), so per-synapse bounds need no stored `w0` array. Zero extra memory.
- **Calibration headroom is ~+8%** ("silent at 0.027, saturated at 0.054"). This
  is why dishabituation had to be a **restore of `d` toward 1.0**, never a gain
  above it. A `SENS_MAX = 1.5` would have saturated the graph on the first
  startle. Both depression pools are bounded by `d ≤ 1`; IP is one-sided (theta
  only ever rises); Hebbian is exactly L1-conserved per presynaptic row, so the
  total current a retina cell delivers is invariant and the DSGC pool's mean
  input cannot move. All four are safe **by construction**, not by tuning.
- **The motion detector is a matched filter at exactly 3 grid cells.**
  `_downsample` point-samples, so `ROWS = linspace(0, 799, 12).astype(int)` and 3
  rows apart is **218 px**; 3 columns is 225 px. A stimulus that translates 218
  px between `retina.step()` calls drives `dsgc_down` to saturation and
  `dsgc_up` to *exactly zero* — the discriminator is the **null in `up`**, not
  the peak in `down`, which clips at `motion_max`. A ball smaller than ~75 px
  falls between sample points and is invisible.

### The measurement that changed the feature

A ball on a plain field does not move this brain. Measured at r = 40, 90 and
150 px: **|dy| ≤ 0.016**, against the 0.08 needed to move anything — and the
Mauthner cell fires on 2–4 frames in 12. A drifting grating gives **0.78**.

```
still grating          |dy| 0.004     0/8  escapes
grating +218 px/frame  |dy| 0.777     0/12 escapes
grating -218 px/frame  |dy| 0.780     0/12 escapes
ball alone (r=150)     |dy| 0.016     4/12 escapes   <- the fish flinches
ball over a grating    |dy| 0.613     0/15 escapes
```

So the table is a **rig**, the kind a real lab builds: the floor is a grating
and `grating_dir()` — one line of Python, and the only line that is not the
fish — points it the way the ball lies. The fish supplies the optomotor reflex
and nothing else. `rig=False` renders the raw ball and is the feature's own
falsification: |dy| falls 0.78 → 0.004 and the paddle stops. The site says all
of this in plain words rather than claiming the fish tracks a ball.

### Cost — and the pattern worth remembering

First cut was **+55.8%** per frame for all four rules, over the ≤35% budget.
Both overruns were the same mistake: *a term that does not depend on anything
the inner loop computes does not belong in a loop that runs 187,053 × 400 = 75
million times a frame.*

- IP's threshold **decay** cost **+26.3%** to move theta by 0.002 mV. Moved to
  `_ip_decay`, once per window → **+9.5%**.
- The slow pool's **recovery** cost **+7.2%** to change `d2` by 0.03% (τ = 600 s
  against a 200 ms window). Moved to `_dep2_recover`, once per window and in
  **closed form**, so the window version is if anything the more accurate of the
  two → **+0.3%**. The *depletion* is per-spike and stayed in `_propagate`.

Final: **all four on, +33.1% per frame** (0.239 s against a ~1 s thought).

### Two things that cost an hour each

- **`retina._flow` was reading a flash on the first frame of every life.** With
  `_prev is None` it substituted `prev = curr`, which does not give zero — it
  gives the *spatial* gradient `|curr[s:] - curr[:-s]|`, as large as real motion
  on any textured image. Only `right` had ever been guarded; the other three
  were not. Fixed, and `reset()` added for when the scene is **cut** rather than
  moved (new page, entering the table) — otherwise frame one of the new scene is
  differenced against the last frame of the old one. Ball-on-plain escapes fell
  4/12 → 2/12 on the fix alone.
- **The synthetic graph's spinal pool is near-critical**, and it nearly got
  blamed on the dep2 refactor. A **0.0006** change in `d2` — four orders of
  magnitude smaller than the thing being measured — swung spinal from 11.2 to
  15.2 Hz. That is a brain calibrated between "silent at 0.027" and "saturated
  at 0.054" behaving as it should, not drift. There is now a comment in
  `_smoke()` saying so, because the obvious next move is to pin an assertion to
  that number and it would fail for reasons nobody can act on.

### Does the learning help? No — and that is the honest answer

30 identical rallies with the Hebbian rule off, then the same 30 with it on:

```
hebbian off -> on:  |dy| 0.5739 -> 0.5843   paddle-ball 91.3 -> 94.7 px
retina weights moved: 0.00e+00 (off) vs 3.61e-02 (on)
difference 3.3 px vs within-run drift 7.0 px -> inside the noise
```

The synapses genuinely changed; the play did not. This is what a reward-free
rule should be expected to do, it is printed on the page as a null rather than
buried, and the fix for it — a reward signal — is exactly what the project
declined to invent. `tools/rally.py --compare` reproduces it;
`tools/rally.py --soak` replays 2,000 frames and re-asserts the bounds.

### Verified
`smoke OK` **and** `smoke OK (plastic)`; `build/calibration.json` still
0.050141; kernel parity numba-vs-numpy **exact** on `d`, `d2`, `theta`,
`last_spike` and `weights` across all six flag combinations (v/isyn ~3e-16,
summation order); per-row L1 conserved to 2.6e-07; habituation train still
habituates with all four on (215 → 120 Hz); endpoints refuse a 3 KB body on the
header without reading it, second joiner gets 409, bad token 403, everything
404s without `ZF_PONG`.

## 7e · The table, played (2026-09-14)

Shipped it, then played a point against the live fish and lost 6–1. The trace is
the whole of round two:

```
  t   ball y   fish y   floor   dy
 1.2     291      400   still   -0.005
 2.4     338      400    down   -0.010      <- an order of magnitude under 0.08
 6.0     498      475    down   -0.756
 7.1     521      560      up   +0.937      <- reversed in one thought
```

`dy` is tracking **the floor**, not the ball, and it does not pretend otherwise.
That is the feature working as designed — but it also meant the fish never
missed, which is the thing the next four changes were about.

### The rig aims less now, and win% was the wrong way to decide

"It's auto-hitting the ball." It was. The fix is a **duty cycle**, not a weaker
push, and the reason is the matched filter: the grating steps exactly one lag
(218 px) or it does not step, and a smaller step lands off the **null in `up`**
and drives both channels. There is no strength to turn down — only *how often*.
A Bresenham accumulator, not an RNG, because the fish is a closed loop and a
random floor is a different experiment every rally.

`tools/rally.py --fair` puts a *tracking* visitor on the other side (the old
`play()` parked it at the centre, which the fish beat 169–73) seeing the ball one
thought late, and sweeps:

```
 aim    |dy|   paddle-ball   fish win%
1.00   0.566      89.5 px      59.5%
0.75   0.435     111.4 px      65.1%
0.50   0.299     122.7 px      48.1%
0.25   0.173     154.1 px      46.0%
0.00   0.005     178.1 px      47.8%
```

**Do not read the win% column.** At aim 0.00 the rig is off, |dy| is 0.005 and
the paddle is effectively still — and the fish *still* takes 47.8%. That column
measures this opponent missing, not the fish playing, and anything tuned against
it would be tuned against noise. The two columns that mean something are
monotonic:

- **paddle-ball must exceed the paddle's own reach**, `PADDLE_H/2 + BALL_R` =
  **101 px**, or the fish is within range of the ball on average and looks like
  it is auto-hitting. That is the whole of "it's auto-hitting the ball", as a
  number. Rules out 1.00; 0.75 only just clears.
- **|dy| must stay well clear of 0.08**, or the paddle stops being visibly the
  fish's doing and the section loses its point.

`AIM_DUTY = 0.5` clears both with margin (122.7 px, 0.299) and `ZF_PONG_AIM`
overrides it. Live confirmation: the same stationary-paddle soak that gave
169–73 now gives **10–9**.

### Three things found by making the ball smooth

The ball used to teleport ~218 px once a thought. It is now cut into
`RALLY_SUBSTEPS = 16` — *physics only*: the grating still steps one lag per
thought and the retina still samples once per thought, so nothing the brain is
shown changed.

- **`Sim.run()` does not tick after the final chunk** (`if done < n_steps:
  tick()`). With `tick_every=25, n_steps=400` you get **15** ticks, not 16 — so
  every thought would have quietly lost a sixteenth of the ball's speed. Paid as
  an explicit remainder `advance(1.0 - spent)` after `run()` returns.
- **Sub-stepping legitimately changes bounce angles**, because a sub-step catches
  the paddle face before the ball has overshot it. The first assertion pinned
  whole-step and sub-step to identical positions and failed; the honest test is
  *free flight only*, plus that the score stays linear in `frac`.
- **Smoothing the frame does not smooth the table.** `_publish_frame` was being
  called at `RALLY_FPS` and the rendered `/frame.jpg` did get smooth — but the
  page draws its table from `state["pong"]`, not from the camera, and the
  heartbeat was still going out once a thought. So *the thing you actually play
  on* was unchanged. Fixed by republishing the heartbeat from the same tick,
  exactly as `_camera_tick` already does for the cursor, and by raising the SSE
  loop's rate cap from `CAM_FPS` to `max(CAM_FPS, RALLY_FPS)` — a cap, not a
  pace, so the browsing path still goes out at 6/s. Measured at a watching
  client over 20 s with the seat held: **the ball moves 22.3 px between
  updates, against ~218 px before** — one matched-filter lag was the whole of
  the old animation.

### The score is the dinosaur's

Fish-vs-visitor was the wrong scoreboard for a fish that is not trying to win.
Chrome's dino counts **distance survived** and keeps a HI, which is exactly the
interesting quantity here: how long the two of you kept the ball alive. Ticks
per sub-step, +100 on a return, resets on a miss. HI lives in the process — a
restart forgets it, and the page says so rather than pretending it persists.
The transcript lines carry the five digits (`… 00415.`), like the dino's.

### Cloudflare blocks `Python-urllib`

An hour, nearly misread as a seat or rate-limit bug: `POST /pong/join` returned
**403 before it ever reached the fish**, while the identical `curl` worked. It
is the default user agent. Any scripted client of `live.zfbrain.online` needs a
browser UA; browsers are unaffected, so nothing on the site ever saw it.

### Verified, round two

`pong smoke OK` with the duty cycle, sub-step and dino-score assertions added;
`tools/rally.py --soak` re-run on the sub-stepped path, **`soak OK`, 8/8**
(worst row L1 4.77e-07 against the 1e-3 gate, theta one-sided at
[0.000, 1.042] mV, mean |dy| 0.297 against the 0.08 gate) — and the same
stationary-paddle soak that read **169–73** before now reads **207–181**, which
is the handicap doing what it was measured to do. Live: joined the restarted
fish, took the point 5–2, transcript `5 returns before the point ended. Mean
decoded drift 0.231; the visitor took it. 00973 — new best.`

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
