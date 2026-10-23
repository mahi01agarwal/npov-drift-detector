#!/usr/bin/env python3
"""Paired bootstrap of the delta (META_FLOOR+neural) - (META_FLOOR).
Overlapping CIs on two specs do NOT test a paired difference; this does, by
resampling pairs and recomputing both out-of-fold AUCs on the SAME resample."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "."; OUT = ROOT + "/npov_data"; SEED = 20261021
rng = np.random.default_rng(SEED)

meta = pd.read_csv(OUT + "/nb2_features.csv"); meta["pageviews_missing"] = meta.pageviews.isna().astype(float)
lex  = pd.read_csv(OUT + "/nb3_lexicon_features.csv")
neu  = pd.read_csv(OUT + "/nb3_neural_features.csv")
FLOOR = ["n_revs","revs_per_day","max_revs_one_day","n_editors","editor_hhi","top_editor_share",
         "anon_share","bot_share","minor_share","reverting_share","reverted_share",
         "new_account_edit_share","size_end","size_net_delta","size_abs_churn","size_delta_std",
         "empty_summary_share","talk_revs"]
NEU = ["neural_mean","neural_p75","neural_max","neural_sub_share","neural_top_mean"]
df = meta.merge(neu[["title","pair","y"]+NEU], on=["title","pair","y"], how="left")
act = df.groupby("pair").empty_window.max(); df = df[df.pair.isin(act[act==0].index)].reset_index(drop=True)
for c in NEU: df[c] = df[c].fillna(0.0)

def oof(cols):
    X = df[cols].to_numpy(float); y = df.y.to_numpy(int); g = df.pair.to_numpy()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    p = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p, y, g

pf, y, g = oof(FLOOR); pn, _, _ = oof(FLOOR + NEU)
a_floor, a_neu = roc_auc_score(y, pf), roc_auc_score(y, pn)
pairs = df.pair.unique(); idx = {p: np.where(g == p)[0] for p in pairs}
deltas = []
for _ in range(5000):
    samp = rng.choice(pairs, len(pairs), replace=True)
    ii = np.concatenate([idx[p] for p in samp])
    try: deltas.append(roc_auc_score(y[ii], pn[ii]) - roc_auc_score(y[ii], pf[ii]))
    except ValueError: pass
deltas = np.array(deltas); lo, hi = np.percentile(deltas, [2.5, 97.5])
print("floor AUC            %.4f" % a_floor)
print("floor+neural AUC     %.4f" % a_neu)
print("paired delta         %+.4f   95%% CI [%+.4f, %+.4f]" % (a_neu - a_floor, lo, hi))
print("P(delta>0)           %.3f" % (deltas > 0).mean())
