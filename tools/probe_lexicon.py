#!/usr/bin/env python
"""Step 0: is there enough in this brain's firing to hang words on?

Run before a line of the word feature ships. The claim under test is narrow and
falsifiable: *the same state gets the same word*. Everything else about the
feature — the drawn panels, the budget, the endpoint — works either way; this
file decides whether the **grown** half of the vocabulary is real or whether we
ship the hand-written anchors alone and say on the page that the unsupervised
half was measured and did not differentiate.

Six gates, and the repo's convention holds: kill or degrade, never tune until it
looks good.

  0a  Noise floor.  One frame held for 200 thoughts, per-channel sigma. This IS
      lexicon.NOISE_FLOOR — measured here, not chosen.
  0b  Span.  Drifting gratings and rendered word panels; per-channel
      across-condition sigma. A channel with across/within < 2 is dropped.
      Fewer than three survivors and the learned half does not ship.
  0c  Ceiling.  PCA on 0b, components above the 0a floor, and the product
      prod(span_k / noise_k). This is the predicted live-word count and it gets
      published before shipping, so the outcome cannot be retrofitted.
  0d  Re-identification — the real gate.  Fit on the first half by time, test on
      the second. Does a condition come back with the same word? Needs >= 3x
      chance AND >= 0.5 absolute on the drifted-text conditions.
  0e  Timer check.  Per channel, R^2 against elapsed sim time vs against
      condition identity. Time wins, channel goes. This is what kills raw
      habituation and raw theta, and it is the likeliest silent failure in the
      whole design: a map that is really a clock, wearing a vocabulary.
  0f  Isolation.  A seeded run with the lexicon on and off must give
      bit-identical rates_hz, and features()+observe() must cost under 0.5 ms
      against a thought that takes about a second.

    python tools/probe_lexicon.py             # everything (~3 min)
    python tools/probe_lexicon.py --quick     # fewer thoughts per condition
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lexicon  # noqa: E402
import show  # noqa: E402
from fishsim import FishSim, load_graph, plasticity_on, set_plasticity  # noqa: E402
from retina import Retina  # noqa: E402

VIEW_W, VIEW_H = 1280, 800
ROWS = np.linspace(0, VIEW_H - 1, 12).astype(int)
DY_PX = int(ROWS[4] - ROWS[1])
GRATING_P = 2.0 * (ROWS[7] - ROWS[1]) / 3.0
MOTION_GAIN = 1.5
BG = 232

_Y = np.arange(VIEW_H, dtype=np.float32)[:, None]

# the words a stranger might actually send, drawn the way show.py draws them
PANELS = ("hope", "cold water", "are you there", "small bright fish")


def grating(phase):
    band = 40.0 + 175.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (_Y - phase) / GRATING_P))
    return np.repeat(band.astype(np.uint8), VIEW_W, axis=1)


def panel(text, pad=0, offset=0):
    """A rendered word panel, optionally a viewport sliding down a taller
    canvas — which is what roam._show will actually do, and the only way a page
    of text makes optic flow after the first frame."""
    img = show.render(text, pad=pad).convert("L")
    arr = np.asarray(img)
    return arr[offset:offset + VIEW_H]


def conditions(quick=False):
    """(label, frames). Every condition is a list of full frames fed in order;
    the drifted-text ones are the conditions 0d is actually judged on, because
    they are what a greeting looks like."""
    n = 4 if quick else 8
    out = [
        ("still grating", [grating(0.0)] * n),
        ("grating down", [grating(k * DY_PX) for k in range(1, n + 1)]),
        ("grating up", [grating(-k * DY_PX) for k in range(1, n + 1)]),
        ("blank page", [np.full((VIEW_H, VIEW_W), BG, np.uint8)] * n),
    ]
    pad = 600
    for text in PANELS:
        frames = [panel(text, pad=pad, offset=int(k * pad / n))
                  for k in range(n)]
        out.append((f"text drift: {text}", frames))
        out.append((f"text still: {text}", [panel(text)] * n))
    return out


TEXT_CONDS = tuple(f"text drift: {t}" for t in PANELS)


def _fresh(graph):
    sim, ret = FishSim(graph), Retina()
    groups = graph["groups"]
    for _ in range(8):                      # settle, the way smoke() does
        out = ret.step(grating(0.0))
        for g, hz in ret.rates(out, motion_gain=MOTION_GAIN).items():
            sim.set_drive(groups.get(g, []), hz)
        sim.run(400)
    return sim, ret


def _run(sim, ret, groups, frames, rest):
    """Feed frames, return one raw feature row per thought plus the sim clock."""
    rows, ts = [], []
    for fr in frames:
        out = ret.step(fr)
        for g, hz in ret.rates(out, motion_gain=MOTION_GAIN).items():
            sim.set_drive(groups.get(g, []), hz)
        detail = sim.run(400)
        for g in ("retina", "nmlf", "vspn", "mauthner", "spinal"):
            hz = detail["rates_hz"].get(g, 0.0)
            rest[g] = hz if g not in rest else 0.95 * rest[g] + 0.05 * hz
        dec = sim.decode(detail["rates_hz"], rest=rest)
        rows.append(lexicon.features(detail, dec, rest))
        ts.append(detail["t"])
    return np.array(rows, dtype=np.float64), np.array(ts)


# ---------------------------------------------------------------------------
# 0a — the noise floor
# ---------------------------------------------------------------------------
def probe_a(graph, n=200):
    print("=" * 72)
    print(f"0a  noise floor: one held frame, {n} thoughts, per-channel sigma")
    sim, ret = _fresh(graph)
    rest = {}
    X, _t = _run(sim, ret, graph["groups"], [grating(0.0)] * n, rest)
    # sigma in the z-space the codebook lives in, not in raw units: normalise
    # by the same running statistics lexicon.encode uses, fitted on this run.
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.maximum(sd, 1e-6)
    within = Z.std(0)
    print()
    print(f"    {'channel':<10} {'raw mean':>10} {'raw sigma':>10}")
    for i, name in enumerate(lexicon.FEATURES):
        print(f"    {name:<10} {mu[i]:>10.4f} {sd[i]:>10.4f}")
    floor = float(np.median(sd / np.maximum(np.abs(mu), 1e-6)))
    print()
    print(f"    held-frame sigma is the floor a split has to clear.")
    print(f"    lexicon.NOISE_FLOOR is in z-units; on this run a held frame has "
          f"z-sigma {within.mean():.3f} by construction (self-normalised), so "
          f"the usable floor comes from 0b's within/across split below.")
    return dict(mu=mu, sd=sd, rel=floor)


# ---------------------------------------------------------------------------
# 0b — span
# ---------------------------------------------------------------------------
def probe_b(graph, quick=False):
    print("=" * 72)
    print("0b  span: per-channel across-condition sigma vs within-condition")
    # ONE continuous sim for every condition, in a seeded interleaved order,
    # repeated. This is not a detail. Running each condition on a fresh sim —
    # which is what the first version of this file did — gives every condition
    # an identical elapsed-time profile, so any monotone channel (theta above
    # all) is *perfectly* predicted by time-within-condition by construction.
    # 0e then "discovers" that theta is a clock no matter what theta is doing.
    # Interleaving decorrelates condition identity from elapsed time, which is
    # the only arrangement in which 0e means anything and the only one in which
    # 0d's split by time is a real held-out test.
    groups = graph["groups"]
    conds = conditions(quick)
    reps = 3 if quick else 6
    sim, ret = _fresh(graph)
    rest = {}
    per_cond, per_t = {label: [] for label, _ in conds}, {label: [] for label, _ in conds}
    rng = np.random.default_rng(20260915)
    t0 = time.time()
    for rep in range(reps):
        order = list(range(len(conds)))
        rng.shuffle(order)
        for j in order:
            label, frames = conds[j]
            X, t = _run(sim, ret, groups, frames, rest)
            per_cond[label].append(X)
            per_t[label].append(t)
        print(f"    rep {rep + 1}/{reps} done ({time.time() - t0:.0f}s)")
    per_cond = {k: np.vstack(v) for k, v in per_cond.items()}
    per_t = {k: np.concatenate(v) for k, v in per_t.items()}
    n_all = sum(len(v) for v in per_cond.values())
    print(f"    {n_all} thoughts across {len(conds)} conditions, one continuous "
          f"life, {reps} interleaved repeats")
    allX = np.vstack(list(per_cond.values()))
    mu, sd = allX.mean(0), np.maximum(allX.std(0), 1e-9)
    Z = {k: (v - mu) / sd for k, v in per_cond.items()}
    means = np.array([v.mean(0) for v in Z.values()])
    across = means.std(0)
    within = np.array([v.std(0) for v in Z.values()]).mean(0)
    print()
    print(f"    {'channel':<10} {'across':>8} {'within':>8} {'ratio':>7}  verdict")
    keep = []
    for i, name in enumerate(lexicon.FEATURES):
        ratio = across[i] / max(within[i], 1e-9)
        ok = ratio >= 2.0
        if ok:
            keep.append(name)
        print(f"    {name:<10} {across[i]:>8.3f} {within[i]:>8.3f} {ratio:>7.2f}  "
              f"{'keep' if ok else 'DROP'}")
    print()
    print(f"    survivors ({len(keep)}): {', '.join(keep) or 'none'}")
    ok = len(keep) >= 3
    print(f"    0b {'PASS' if ok else 'FAIL'} (needs >= 3 channels)")
    return ok, dict(Z=Z, t=per_t, keep=keep, across=across, within=within,
                    mu=mu, sd=sd)


# ---------------------------------------------------------------------------
# 0c — the ceiling, published before the result exists
# ---------------------------------------------------------------------------
def probe_c(res_b):
    print("=" * 72)
    print("0c  ceiling: how many words could honestly be distinguished")
    Z = res_b["Z"]
    keep_idx = [lexicon.FEATURES.index(k) for k in res_b["keep"]]
    if not keep_idx:
        print("    no surviving channels — predicted live words: 0")
        return 0
    allZ = np.vstack([v[:, keep_idx] for v in Z.values()])
    allZ = allZ - allZ.mean(0)
    # within-condition scatter is the noise; across-condition is the signal
    noise = np.vstack([v[:, keep_idx] - v[:, keep_idx].mean(0) for v in Z.values()])
    _u, s_sig, _v = np.linalg.svd(allZ, full_matrices=False)
    _u, s_noi, _v = np.linalg.svd(noise, full_matrices=False)
    n = allZ.shape[0]
    sig = s_sig / np.sqrt(max(1, n))
    noi = s_noi / np.sqrt(max(1, n))
    live = 1.0
    used = 0
    print(f"    {'component':>9} {'signal':>8} {'noise':>8} {'ratio':>7}")
    for k in range(len(sig)):
        ratio = sig[k] / max(noi[k], 1e-9)
        print(f"    {k:>9} {sig[k]:>8.3f} {noi[k]:>8.3f} {ratio:>7.2f}")
        if ratio > 1.0:
            live *= ratio
            used += 1
    ceiling = float(lexicon.LIST_SIZE - 58)
    print()
    print(f"    {used} components above the floor; prod(span/noise) = {live:,.0f}")
    print(f"    PREDICTION, recorded before the feature runs on a live fish:")
    if live >= ceiling:
        # An honest non-prediction. The product rule assumes every component is
        # independently resolvable across its whole span, which is optimistic;
        # when it lands above the word list it has told us nothing except that
        # the binding constraint is somewhere else — SPLIT_N, and how much of
        # this space a fish reading web pages ever actually visits.
        print(f"    the resolvable-cell estimate ({live:,.0f}) is larger than "
              f"the {ceiling:.0f} words available, so 0c sets no useful upper "
              f"bound. What binds is SPLIT_N ({lexicon.SPLIT_N} thoughts per "
              f"split) and how much of this space the fish visits, and the")
        print(f"    site reports whatever the live count turns out to be.")
    else:
        print(f"    the grown half should settle near {live:,.0f} words in use "
              f"(ceiling {ceiling:.0f}).")
    return live


# ---------------------------------------------------------------------------
# 0d — re-identification, the real gate
# ---------------------------------------------------------------------------
def probe_d(res_b):
    print("=" * 72)
    print("0d  re-identification: fit on the first half by time, test on the "
          "second")
    Z = res_b["Z"]
    halves = {k: len(v) // 2 for k, v in Z.items()}
    # Channel selection happens HERE, on the fit half only, and not from 0b —
    # 0b ranked channels using all the data, and a codebook built on channels
    # chosen with the test half in hand would be scored against its own answer
    # key. Same rule as 0b (across/within >= 2), different data.
    fit_rows = {k: v[:halves[k]] for k, v in Z.items()}
    f_means = np.array([v.mean(0) for v in fit_rows.values()])
    f_across = f_means.std(0)
    f_within = np.array([v.std(0) for v in fit_rows.values()]).mean(0)
    keep_idx = [i for i in range(lexicon.NDIM)
                if f_across[i] / max(f_within[i], 1e-9) >= 2.0]
    keep_names = [lexicon.FEATURES[i] for i in keep_idx]
    print(f"    channels chosen on the fit half alone ({len(keep_idx)}): "
          f"{', '.join(keep_names) or 'none'}")
    if len(keep_idx) < 3:
        print("    0d FAIL — fewer than 3 channels survive on the fit half")
        return False, 0.0
    # Quantise on those channels only. Zeroing the rest is the same thing as
    # dropping them for a Euclidean distance, and keeps FEATURES 12 wide so the
    # vector stays readable against the published table. The first version of
    # this file selected channels and then quantised on all twelve anyway,
    # which meant 0d was scored on a distance half made of the noise 0b had
    # just condemned — a strawman, and it failed for the wrong reason.
    mask = np.zeros(lexicon.NDIM)
    mask[keep_idx] = 1.0
    Z = {k: v * mask for k, v in Z.items()}
    lex = lexicon.Lexicon(path="/tmp/probe_lexicon_fit.npz", learn=True)
    lex.proto = np.zeros((0, lexicon.NDIM), dtype=np.float32)
    lex.count = np.zeros(0, dtype=np.int64)
    lex.total = np.zeros(0, dtype=np.int64)
    lex.scatter = np.zeros((0, lexicon.NDIM, lexicon.NDIM))
    lex.cell_word = []
    # fit: first half of every condition, interleaved so no cell is a clock
    order = []
    for label, v in Z.items():
        for r in range(halves[label]):
            order.append((label, v[r]))
    # SPLIT_N is sized for a fish that thinks for days; against a few hundred
    # probe thoughts it guarantees one cell, and one cell scores 1.000 here
    # because every condition trivially "comes back" as the only word there is.
    # A gate that cannot fail is not a gate. So the probe scales the threshold
    # to its own data volume, states both numbers, and 0d below is scored
    # against the fitted word marginal rather than against 1/n_conditions.
    split_n = max(12, len(order) // (2 * len(Z)))
    old_split = lexicon.SPLIT_N
    lexicon.SPLIT_N = split_n
    try:
        for _label, row in order:
            # _learn, not observe: these rows were z-scored on line 141 by the
            # batch statistics. Running them through encode() again normalises
            # twice, and since the warm-up fix it would also feed the codebook
            # 200 zero vectors before the first real one. The probe owns the
            # scale here — the codebook is the only thing under test.
            lex._learn(row)
    finally:
        lexicon.SPLIT_N = old_split
    print(f"    fitted on {len(order)} thoughts -> {len(lex.proto)} cells, "
          f"{lex.splits} splits (probe SPLIT_N {split_n}, live {old_split})")
    if len(lex.proto) < 2:
        print("    0d FAIL — the codebook never split, so re-identification is "
              "vacuous: there is only one word to come back as.")
        return False, 0.0
    # test: second half, frozen
    got, total = {}, {}
    fit_word = {}
    for label, v in Z.items():
        ws = [lex._quantise_z(v[r])[1] for r in range(halves[label])]
        fit_word[label] = max(set(ws), key=ws.count) if ws else None
    hits = 0
    n = 0
    text_hits, text_n = 0, 0
    marginal = {}
    for label, v in Z.items():
        for r in range(halves[label], len(v)):
            w = lex._quantise_z(v[r])[1]   # already z, see above
            marginal[w] = marginal.get(w, 0) + 1
            n += 1
            ok = (w == fit_word[label])
            hits += ok
            if label in TEXT_CONDS:
                text_n += 1
                text_hits += ok
        got[label] = fit_word[label]
        total[label] = len(v) - halves[label]
    # Chance is not 1/n_conditions. If the quantiser dumps 90% of thoughts into
    # one cell, guessing that cell every time already scores 0.9, and a gate
    # scored against 1/12 would wave it through. The baseline is the rate a
    # predictor gets by ignoring the condition and drawing from the word
    # marginal the test half actually produced.
    chance = sum((c / max(1, n)) ** 2 for c in marginal.values())
    rate = hits / max(1, n)
    trate = text_hits / max(1, text_n)
    print()
    for label in Z:
        print(f"    {label:<26} -> {got[label]}")
    print()
    print(f"    test-half words in use: {len(marginal)}; "
          f"largest cell holds {max(marginal.values())/max(1,n):.0%}")
    print(f"    re-identification {rate:.3f} over all conditions "
          f"(marginal chance {chance:.3f}, {rate/max(chance,1e-9):.1f}x)")
    print(f"    on drifted text        {trate:.3f}")
    ok = rate >= 3 * chance and trate >= 0.5
    print(f"    0d {'PASS' if ok else 'FAIL'} (needs >= 3x chance AND >= 0.5 on "
          f"drifted text)")
    if not ok:
        print("    -> the headline claim is false on this data. Ship the "
              "anchored map only, and say so on the page.")
    return ok, rate


# ---------------------------------------------------------------------------
# 0e — is it a clock?
# ---------------------------------------------------------------------------
def probe_e(res_b):
    print("=" * 72)
    print("0e  timer check: R^2 against elapsed sim time vs condition identity")
    Z, T = res_b["Z"], res_b["t"]
    labels = list(Z)
    y_time = np.concatenate([T[k] - T[k][0] for k in labels])
    X = np.vstack([Z[k] for k in labels])
    # condition identity as one-hot; R^2 from a least-squares fit of each
    onehot = np.zeros((len(X), len(labels)))
    o = 0
    for j, k in enumerate(labels):
        onehot[o:o + len(Z[k]), j] = 1.0
        o += len(Z[k])

    def r2(design, y):
        d = np.column_stack([design, np.ones(len(design))])
        beta, *_ = np.linalg.lstsq(d, y, rcond=None)
        resid = y - d @ beta
        ss = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float((resid ** 2).sum()) / max(ss, 1e-12)

    print()
    print(f"    {'channel':<10} {'R2(time)':>9} {'R2(cond)':>9}  verdict")
    keep = []
    for i, name in enumerate(lexicon.FEATURES):
        rt = r2(y_time[:, None], X[:, i])
        rc = r2(onehot, X[:, i])
        ok = rc > rt
        if ok:
            keep.append(name)
        print(f"    {name:<10} {rt:>9.3f} {rc:>9.3f}  "
              f"{'keep' if ok else 'DROP (a clock)'}")
    print()
    print(f"    condition beats time on {len(keep)}/{len(lexicon.FEATURES)}: "
          f"{', '.join(keep)}")
    hp = [n for n in ("hab_hp", "theta_hp") if n in keep]
    print(f"    the high-passed slow channels that survived: {hp or 'none'}")
    ok = len(keep) >= 3
    print(f"    0e {'PASS' if ok else 'FAIL'}")
    return ok, keep


# ---------------------------------------------------------------------------
# 0f — isolation: the brain must not notice
# ---------------------------------------------------------------------------
def probe_f(graph, n=12):
    print("=" * 72)
    print("0f  isolation: bit-identical rates with the lexicon on and off")
    groups = graph["groups"]
    frames = [grating(k * DY_PX) for k in range(1, n + 1)]

    def go(with_lex):
        np.random.seed(20260915)
        sim, ret = FishSim(graph), Retina()
        lex = lexicon.Lexicon(path="/tmp/probe_lexicon_iso.npz") if with_lex else None
        rest, out, cost = {}, [], []
        for fr in frames:
            o = ret.step(fr)
            for g, hz in ret.rates(o, motion_gain=MOTION_GAIN).items():
                sim.set_drive(groups.get(g, []), hz)
            detail = sim.run(400)
            dec = sim.decode(detail["rates_hz"], rest=rest)
            if lex is not None:
                t0 = time.perf_counter()
                v = lexicon.features(detail, dec, rest)
                lex.observe(v)
                lex.utter(v, dec)
                cost.append((time.perf_counter() - t0) * 1000.0)
            out.append(tuple(sorted(detail["rates_hz"].items())))
        return out, cost

    a, _ = go(False)
    b, cost = go(True)
    same = a == b
    ms = float(np.mean(cost)) if cost else 0.0
    worst = float(np.max(cost)) if cost else 0.0
    print(f"    rates_hz identical across {n} thoughts: {same}")
    print(f"    features()+observe()+utter(): mean {ms:.3f} ms, worst "
          f"{worst:.3f} ms (budget 0.5 ms against a ~1 s thought)")
    ok = same and worst < 0.5
    print(f"    0f {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="build/graph.npz")
    ap.add_argument("--groups", default="build/groups.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plastic", action="store_true",
                    help="measure the calibrated brain instead of the live one")
    args = ap.parse_args()

    # The four slow rules default OFF in code so `python fishsim.py` reproduces
    # the calibration numbers, and .env turns them on for the fish that is
    # actually running. Measuring with them off was this file's first real bug:
    # `intrinsic` reports a flat 0.0 when ZF_IP is unset, so theta_hp came back
    # with exactly zero variance and 0b/0e "measured" a channel that was not
    # there. A probe that decides a headline claim has to be pointed at the
    # brain the claim is about.
    if not args.no_plastic:
        set_plasticity(ip=1, dep2=1, sens=1, hebb=1)
    print(f"plasticity: {', '.join(plasticity_on()) or 'none'}")

    t0 = time.time()
    graph = load_graph(args.graph, args.groups)
    print(f"graph: {graph['meta'].get('label')} "
          f"{graph['meta'].get('neurons'):,} neurons, {time.time()-t0:.1f}s")
    probe_a(graph, n=60 if args.quick else 200)
    ok_b, res_b = probe_b(graph, quick=args.quick)
    live = probe_c(res_b)
    ok_d, _rate = probe_d(res_b) if ok_b else (False, 0.0)
    ok_e, _keep = probe_e(res_b)
    ok_f = probe_f(graph)
    print("=" * 72)
    print(f"0b {'PASS' if ok_b else 'FAIL'}   0d {'PASS' if ok_d else 'FAIL'}   "
          f"0e {'PASS' if ok_e else 'FAIL'}   0f {'PASS' if ok_f else 'FAIL'}")
    print(f"0c predicted live words: {live:.0f}")
    if ok_b and ok_d and ok_e and ok_f:
        print("-> ship both halves.")
    else:
        print("-> ship the anchored map only, and say on the page that the "
              "unsupervised half was measured and did not differentiate.")


if __name__ == "__main__":
    main()
