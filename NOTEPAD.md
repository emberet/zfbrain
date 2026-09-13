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
- `HOME_SEEDS`' own comment says "no captcha-walled sites here". That was true
  when written. Four of them have walled since, so the list wants re-checking
  rather than trusting the comment.

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
