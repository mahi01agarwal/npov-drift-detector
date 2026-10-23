#!/usr/bin/env python3
"""
Notebook 4b, stage C (no GPU)  —  due-weight-by-VOLUME lead-time curve.

NB4 showed whole-article subjectivity is a flat, elevated, STATIC trait (no drift).
Before paying for the GPU per-section subjectivity run, this asks the cheapest
section-level question: does CONTENT PILE INTO ONE SECTION in the run-up to the tag?
That is a volume proxy for a due-weight problem (one angle crowding out the rest).

Signal, per (article, lag), from section byte sizes (prose sections only):
  size_hhi      Herfindahl of section sizes   (1 = all in one section, ->0 = many equal)
  max_sec_frac  largest section's share of prose bytes
  size_gini     Gini of section sizes
  n_sections    number of prose sections      (drift could also be FEWER, fatter sections)

Two readings, mirroring NB4 so results drop straight into the paper:
  (A) MATCHED, per lag — pairwise win-rate (tagged side more concentrated than its twin)
      + mean gap, pair-bootstrap 95% CI. Is concentration a static tagged-vs-control trait?
  (B) WITHIN-ARTICLE drift — for the TAGGED side only, on the balanced panel, the paired
      change from lag 365d -> 0d. Does a section actually fatten AS the tag approaches?
      This is the direct drift test; NB4's within-article level barely moved (std 0.006).

Input:  npov_data/nb4_section_sizes.csv   (from build_sections.py, no GPU)
Output: npov_data/nb4_sections_size_curve.csv
"""
import os, numpy as np, pandas as pd

ROOT = "."; OUT = ROOT + "/npov_data"
SEED = 20261022; rng = np.random.default_rng(SEED)
FEATS = ["size_hhi", "max_sec_frac", "size_gini", "n_sections"]

d = pd.read_csv(OUT + "/nb4_section_sizes.csv")
d = d[d.revid.notna() & d.n_sections.gt(0)].copy()   # a real snapshot with prose sections
lags = sorted(d.lag.unique())
print("rows %d | lags %s | pairs %d" % (len(d), lags, d.pair.nunique()))

# ---- wide table per feature: pos/ctl side by side at each (pair, lag) ----------
def wide(df, feat):
    pos = df[df.y == 1][["pair", "lag", feat]].rename(columns={feat: "pos"})
    ctl = df[df.y == 0][["pair", "lag", feat]].rename(columns={feat: "ctl"})
    return pos.merge(ctl, on=["pair", "lag"], how="inner").dropna()

def winrate_gap(sub):
    diff = sub.pos.values - sub.ctl.values
    win  = (diff > 0).mean() + 0.5 * (diff == 0).mean()
    return win, diff.mean()

def boot_pair(sub, fn, n=5000):
    pairs = sub.pair.values
    idx = {p: np.where(pairs == p)[0] for p in np.unique(pairs)}
    up = np.unique(pairs)
    vals = []
    for _ in range(n):
        s = rng.choice(up, len(up), replace=True)
        ii = np.concatenate([idx[p] for p in s])
        vals.append(fn(sub.iloc[ii]))
    return np.array(vals)

rows = []
for feat in FEATS:
    W = wide(d, feat)
    cnt = W.groupby("pair").lag.nunique()
    balanced = set(cnt[cnt == len(lags)].index)

    # (A) matched win-rate per lag --------------------------------------------
    for panel, keep in [("balanced", balanced), ("surviving", None)]:
        for lag in lags:
            sub = W[W.lag == lag]
            if keep is not None:
                sub = sub[sub.pair.isin(keep)]
            if len(sub) < 5:
                continue
            wr, gp = winrate_gap(sub)
            bw = boot_pair(sub, lambda s: winrate_gap(s)[0])
            bg = boot_pair(sub, lambda s: winrate_gap(s)[1])
            rows.append({"feat": feat, "test": "matched_winrate", "panel": panel,
                         "lag": lag, "n_pairs": len(sub), "winrate": wr,
                         "wr_lo": np.percentile(bw, 2.5), "wr_hi": np.percentile(bw, 97.5),
                         "mean_gap": gp,
                         "gap_lo": np.percentile(bg, 2.5), "gap_hi": np.percentile(bg, 97.5)})

    # (B) within-article drift as DIFFERENCE-IN-DIFFERENCES, balanced panel -----
    # A tagged article growing over a year is not drift if its twin grew the same.
    # The drift test is: did the TAGGED side's concentration change MORE than its
    # matched CONTROL's, over the identical 365d -> 0d window?
    far, near = max(lags), min(lags)   # 365d -> 0d
    def side_delta(yv):
        t = d[(d.y == yv) & d.pair.isin(balanced)][["pair", "lag", feat]]
        piv = t.pivot_table(index="pair", columns="lag", values=feat, aggfunc="first")
        if far not in piv.columns or near not in piv.columns:
            return None
        return (piv[near] - piv[far]).rename("d")           # indexed by pair
    dp, dc = side_delta(1), side_delta(0)
    if dp is not None and dc is not None:
        both_p = dp.dropna().index.intersection(dc.dropna().index)
        did = (dp.loc[both_p] - dc.loc[both_p]).values      # >0 = tagged concentrated MORE than twin
        n = len(did)
        rosewin = (did > 0).mean() + 0.5 * (did == 0).mean()
        bd = np.array([did[rng.integers(0, n, n)].mean() for _ in range(5000)])
        bw = np.array([(lambda x: (x > 0).mean() + 0.5*(x == 0).mean())(did[rng.integers(0, n, n)])
                       for _ in range(5000)])
        rows.append({"feat": feat, "test": "within_drift_DiD_365to0", "panel": "balanced",
                     "lag": -1, "n_pairs": n,
                     "winrate": rosewin, "wr_lo": np.percentile(bw, 2.5), "wr_hi": np.percentile(bw, 97.5),
                     "mean_gap": float(did.mean()),
                     "gap_lo": np.percentile(bd, 2.5), "gap_hi": np.percentile(bd, 97.5)})

res = pd.DataFrame(rows)
res.to_csv(OUT + "/nb4_sections_size_curve.csv", index=False)

# ---- report -------------------------------------------------------------------
for feat in FEATS:
    print("\n==================  %s  ==================" % feat)
    sub = res[(res.feat == feat) & (res.test == "matched_winrate") & (res.panel == "balanced")]
    if len(sub):
        print("[A] MATCHED win-rate (tagged side more concentrated than twin), balanced panel")
        print("  %5s %7s %9s %22s %12s" % ("lag", "pairs", "winrate", "95% CI", "mean_gap"))
        for r in sub.sort_values("lag").itertuples(index=False):
            star = " *" if r.wr_lo > 0.5 else ""
            print("  %4dd %7d %9.3f   [%.3f, %.3f]   %+.5f%s"
                  % (r.lag, r.n_pairs, r.winrate, r.wr_lo, r.wr_hi, r.mean_gap, star))
    w = res[(res.feat == feat) & (res.test == "within_drift_DiD_365to0")]
    if len(w):
        r = w.iloc[0]
        rose = "DRIFT: tagged rose MORE" if r.gap_lo > 0 else \
               ("tagged rose LESS" if r.gap_hi < 0 else "NO DRIFT (CI spans 0)")
        print("[B] DiD  365d -> 0d, tagged change minus twin's change (n=%d):"
              "  %+.5f  [%.5f, %.5f]  => %s"
              % (r.n_pairs, r.mean_gap, r.gap_lo, r.gap_hi, rose))
        print("    (%.1f%% of pairs: tagged concentrated more than its twin over the year)"
              % (100 * r.winrate))

print("\nwrote nb4_sections_size_curve.csv")
print("READ: [A] tells you if concentration is a static tagged trait; [B] tells you if it DRIFTS.")
