#!/usr/bin/env python3
"""
Notebook 4, stage C  —  the lead-time curve.

Does the tagged article read as more biased than its matched twin BEFORE a human
tagged it?  Using the matched design directly (no per-lag model training, which would
overfit at the small lags): at each lead time we report, for the neural subjectivity
score, the PAIRWISE WIN-RATE (fraction of pairs where the tagged side scores higher —
this is AUC for a matched feature) and the MEAN GAP, each with a pair-bootstrap 95% CI.

Two panels:
  surviving  — every pair where both sides existed at that lead time (max n, but the
               set of pairs changes with lead time as young articles drop out).
  balanced   — only pairs present at ALL lead times, so the trajectory is not an
               artifact of changing composition. This is the one to read.

Input:  npov_data/nb4_neural_features.csv   (lag-tagged, from score_bias_gpu.py)
Output: npov_data/nb4_leadtime_curve.csv  and  npov_data/nb4_leadtime.png
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."; OUT = ROOT + "/npov_data"
SEED = 20261021; rng = np.random.default_rng(SEED)
FEAT = "neural_sub_share"        # headline: share of sentences the model calls subjective

d = pd.read_csv(OUT + "/nb4_neural_features.csv")
d = d[d.neural_n_scored > 0].copy()           # a real snapshot existed
lags = sorted(d.lag.unique())
print("rows %d | lags %s" % (len(d), lags))

# wide table: one row per (pair, lag) with pos/control scores side by side
def wide(df):
    pos = df[df.y == 1][["pair", "lag", FEAT]].rename(columns={FEAT: "pos"})
    ctl = df[df.y == 0][["pair", "lag", FEAT]].rename(columns={FEAT: "ctl"})
    return pos.merge(ctl, on=["pair", "lag"], how="inner")
W = wide(d)

# balanced panel: pairs present (both sides) at EVERY lag
cnt = W.groupby("pair").lag.nunique()
balanced_pairs = set(cnt[cnt == len(lags)].index)
print("balanced panel: %d pairs present at all %d lags" % (len(balanced_pairs), len(lags)))

def winrate_gap(sub):
    """sub has columns pos, ctl (one row per pair). Returns winrate, meangap."""
    diff = sub.pos.values - sub.ctl.values
    win  = (diff > 0).mean() + 0.5 * (diff == 0).mean()
    return win, diff.mean()

def boot(sub, n=5000):
    pairs = sub.pair.values
    idx = {p: np.where(pairs == p)[0] for p in np.unique(pairs)}
    up = np.unique(pairs)
    wr, gp = [], []
    for _ in range(n):
        s = rng.choice(up, len(up), replace=True)
        ii = np.concatenate([idx[p] for p in s])
        d2 = sub.iloc[ii]
        w, g = winrate_gap(d2); wr.append(w); gp.append(g)
    return (np.percentile(wr, [2.5, 97.5]), np.percentile(gp, [2.5, 97.5]))

rows = []
for panel, keep in [("surviving", None), ("balanced", balanced_pairs)]:
    for lag in lags:
        sub = W[W.lag == lag]
        if keep is not None:
            sub = sub[sub.pair.isin(keep)]
        if len(sub) < 5:
            continue
        wr, gp = winrate_gap(sub)
        (wlo, whi), (glo, ghi) = boot(sub)
        rows.append({"panel": panel, "lag": lag, "n_pairs": len(sub),
                     "winrate": wr, "wr_lo": wlo, "wr_hi": whi,
                     "mean_gap": gp, "gap_lo": glo, "gap_hi": ghi})
res = pd.DataFrame(rows)
res.to_csv(OUT + "/nb4_leadtime_curve.csv", index=False)

print("\nLEAD-TIME CURVE  (feature = %s)" % FEAT)
print("in %% of pairs the TAGGED article already reads as more subjective than its twin:")
for panel in ["balanced", "surviving"]:
    print("\n  [%s panel]" % panel)
    print("  %5s %7s %9s %20s %9s" % ("lag", "pairs", "winrate", "95% CI", "mean_gap"))
    for r in res[res.panel == panel].itertuples(index=False):
        star = " *" if r.wr_lo > 0.5 else ""      # win-rate significantly above chance
        print("  %4dd %7d %9.3f   [%.3f, %.3f]   %+.4f%s"
              % (r.lag, r.n_pairs, r.winrate, r.wr_lo, r.wr_hi, r.mean_gap, star))

# --- figure: balanced-panel win-rate vs lead time ------------------------------
b = res[res.panel == "balanced"].sort_values("lag")
if len(b):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.axhline(0.5, color="#999", lw=1, ls="--", label="chance (0.5)")
    ax.fill_between(b.lag, b.wr_lo, b.wr_hi, color="#3366cc", alpha=0.18)
    ax.plot(b.lag, b.winrate, "o-", color="#3366cc", lw=2, label="tagged > control win-rate")
    ax.set_xlabel("days before the human tag (lead time)")
    ax.set_ylabel("pairwise win-rate  (= matched AUC)")
    ax.set_title("Lead time of the neural subjectivity signal (balanced panel, n=%d)"
                 % int(b.n_pairs.iloc[0]))
    ax.invert_xaxis()                              # onset on the right, earlier to the left
    ax.set_ylim(0.45, max(0.75, b.wr_hi.max() + 0.03))
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT + "/nb4_leadtime.png", dpi=130)
    print("\nwrote nb4_leadtime_curve.csv and nb4_leadtime.png")
