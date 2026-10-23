#!/usr/bin/env python3
"""
Notebook 6  —  precision@k watchlist simulation.  The "so what".

NB4/NB4b showed the text signal is a STATIC RISK FACTOR, not a drift detector. The
actionable form of that finding is a TRIAGE tool: rank articles a year BEFORE any human
tag and the future-tagged ones should concentrate at the top. This measures how well.

Honest watchlist design (two traps avoided):
  * Base rate. Our matched set is 50% positives, so raw precision@k is meaningless for a
    watchlist. We simulate a realistic population where NPOV tags are RARE (prevalence r),
    reweighting from the empirical score distributions, and report LIFT = precision / r.
  * Hard negatives. Our controls are topically MATCHED (same category, similar size) — far
    harder to tell apart than a random article. So these numbers are a CONSERVATIVE lower
    bound; a real population of mostly-obviously-neutral articles would rank easier.

Ranking signal is a single interpretable feature measured at a snapshot (no training, no
peeking): headline lag = onset-365d (the forward claim), with 0d (concurrent) alongside.
Higher subjective score => higher dispute risk => ranked first.

Monte Carlo: for population size N and prevalence r, draw n_pos~Binom(N,r) article scores
from the positive pool and N-n_pos from the negative pool (empirical, with replacement),
rank descending, and read precision/recall in the top-k. B resamples give 95% CIs.

Input:  npov_data/nb4_section_neural.csv   (lag-tagged per-article subjectivity features)
Output: npov_data/nb6_watchlist.csv
"""
import numpy as np, pandas as pd

ROOT = "."; OUT = ROOT + "/npov_data"
SEED = 20261023; rng = np.random.default_rng(SEED)

FEATS       = ["sec_worst_share", "whole_sub_share"]   # strongest static feature + NB4 whole-article
LAGS        = [365, 0]                                  # forward (headline) and concurrent
PREVALENCES = [0.01, 0.02, 0.05]                        # realistic NPOV rarity
KFRACS      = [0.01, 0.02, 0.05, 0.10]                  # top-k as a fraction of the population
N_POP       = 5000
B           = 2000

d = pd.read_csv(OUT + "/nb4_section_neural.csv")
d = d[d.revid.notna()].copy()

def unpaired_auc(pos, neg):
    """P(random positive ranked above random negative) via Mann-Whitney."""
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    rpos = r[:len(pos)].sum()
    return (rpos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

rows = []
for feat in FEATS:
    for lag in LAGS:
        sub = d[d.lag == lag]
        pos = sub[sub.y == 1][feat].dropna().values
        neg = sub[sub.y == 0][feat].dropna().values
        if len(pos) < 20 or len(neg) < 20:
            continue
        auc = unpaired_auc(pos, neg)
        for r in PREVALENCES:
            # pre-draw all B populations
            prec = {k: [] for k in KFRACS}; rec = {k: [] for k in KFRACS}
            for _ in range(B):
                npos = rng.binomial(N_POP, r)
                npos = min(max(npos, 1), N_POP - 1)
                s = np.empty(N_POP); lab = np.zeros(N_POP, dtype=bool)
                s[:npos]  = rng.choice(pos, npos, replace=True); lab[:npos] = True
                s[npos:]  = rng.choice(neg, N_POP - npos, replace=True)
                order = np.argsort(-s)                 # highest score first
                labo = lab[order]
                for kf in KFRACS:
                    kk = max(1, int(round(N_POP * kf)))
                    hits = labo[:kk].sum()
                    prec[kf].append(hits / kk)
                    rec[kf].append(hits / npos)
            for kf in KFRACS:
                pv = np.array(prec[kf]); rv = np.array(rec[kf])
                rows.append({
                    "feat": feat, "lag": lag, "unpaired_auc": auc, "prevalence": r,
                    "top_k_frac": kf, "precision": pv.mean(),
                    "prec_lo": np.percentile(pv, 2.5), "prec_hi": np.percentile(pv, 97.5),
                    "lift": pv.mean() / r, "recall": rv.mean()})

res = pd.DataFrame(rows)
res.to_csv(OUT + "/nb6_watchlist.csv", index=False)

for feat in FEATS:
    for lag in LAGS:
        sb = res[(res.feat == feat) & (res.lag == lag)]
        if sb.empty:
            continue
        auc = sb.unpaired_auc.iloc[0]
        when = "onset-%dd (forward)" % lag if lag else "onset (concurrent)"
        print("\n================  %s  |  ranked at %s  |  unpaired AUC %.3f  ================"
              % (feat, when, auc))
        print("  rank the top-k%% of a simulated population; how many are future-tagged?")
        print("  %5s %6s %10s %22s %7s %8s" % ("prev", "top-k", "precision", "95% CI", "lift", "recall"))
        for r in sb.sort_values(["prevalence", "top_k_frac"]).itertuples(index=False):
            print("  %4.0f%% %5.0f%% %10.3f   [%.3f, %.3f]   %5.1fx %7.2f"
                  % (r.prevalence*100, r.top_k_frac*100, r.precision,
                     r.prec_lo, r.prec_hi, r.lift, r.recall))

print("\nwrote nb6_watchlist.csv")
print("READ: lift = precision / base-rate. lift>1 means the watchlist beats random triage.")
print("      Numbers are a CONSERVATIVE floor (controls are matched/hard negatives).")
