#!/usr/bin/env python3
"""
Notebook 3, stage C  —  does the TEXT beat the 0.76 metadata floor?

CPU, seconds. Joins three feature sets on the same 421 both-active pairs Notebook 2
used, and scores each specification with the SAME harness (logistic + HistGradientBoosting,
pair-grouped 5-fold CV, pair-bootstrap 95% CI). Every content result is reported as a
DELTA over the metadata floor recomputed here on these exact pairs.

Inputs (npov_data/):
  nb2_features.csv          metadata (the floor)         [required]
  nb3_lexicon_features.csv  transparent lexicon rates    [required, stage A]
  nb3_neural_features.csv   GPU subjectivity scores       [optional, stage B]

Output:
  nb3_specification_ladder.csv  and a printed ladder with deltas.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = "."
OUT  = ROOT + "/npov_data"
SEED = 20261021
rng  = np.random.default_rng(SEED)

meta = pd.read_csv(OUT + "/nb2_features.csv")
lex  = pd.read_csv(OUT + "/nb3_lexicon_features.csv")
neural_path = OUT + "/nb3_neural_features.csv"
HAVE_NEURAL = os.path.exists(neural_path)
neu = pd.read_csv(neural_path) if HAVE_NEURAL else None
print("metadata rows %d | lexicon rows %d | neural %s"
      % (len(meta), len(lex), "present" if HAVE_NEURAL else "ABSENT (run stage B)"))

# --- reconstruct NB2's modelling view ------------------------------------------
meta["pageviews_missing"] = meta["pageviews"].isna().astype(float)
META_ALL = ["n_revs","revs_per_day","max_revs_one_day","n_editors","editor_hhi",
            "top_editor_share","anon_share","bot_share","minor_share","reverting_share",
            "reverted_share","new_account_edit_share","size_end","size_net_delta",
            "size_abs_churn","size_delta_std","empty_summary_share","talk_revs",
            "pageviews","empty_window","pageviews_missing"]
# the floor = NB2 spec C: drop traffic and the control-screening artifact
META_FLOOR = [c for c in META_ALL if c not in ("pageviews","pageviews_missing","empty_window")]

# --- join all three on (title, pair, y) ----------------------------------------
LEX = ["peacock_rate","weasel_rate","hedge_rate","factive_rate","assertive_rate",
       "bias_lex_rate","subjective_sent_share","mean_sent_len"]
df = meta.merge(lex[["title","pair","y"] + LEX], on=["title","pair","y"], how="left")
NEU = []
if HAVE_NEURAL:
    NEU = ["neural_mean","neural_p75","neural_max","neural_sub_share","neural_top_mean"]
    df = df.merge(neu[["title","pair","y"] + NEU], on=["title","pair","y"], how="left")

# --- restrict to the 421 both-active pairs (both sides edited in the window) ----
act = df.groupby("pair")["empty_window"].max()          # 0 only if neither side dormant
both_active = act[act == 0].index
df = df[df.pair.isin(both_active)].reset_index(drop=True)
print("both-active pairs: %d (rows %d)" % (df.pair.nunique(), len(df)))

# --- class-dependent-missingness gate on the NEW text features -----------------
# Same lesson as NB2: a feature whose *missingness* tracks the label is a fake signal.
print("\nmissingness-by-class gate (text features):")
bad = []
for c in LEX + NEU:
    m = df.groupby("y")[c].apply(lambda s: s.isna().mean())
    gap = abs(m.get(1, 0) - m.get(0, 0))
    flag = "  <-- FAIL >10pp" if gap > 0.10 else ""
    if gap > 0.10:
        bad.append(c)
    print("  %-22s miss0=%.3f miss1=%.3f gap=%.3f%s" % (c, m.get(0,0), m.get(1,0), gap, flag))
if bad:
    print("WARNING dropping features that fail the gate:", bad)
    LEX = [c for c in LEX if c not in bad]
    NEU = [c for c in NEU if c not in bad]

# text NaNs (article had no usable sentences) -> 0 rate after the gate confirms it's
# not class-dependent; HGB handles NaN natively, logistic needs a fill.
for c in LEX + NEU:
    df[c] = df[c].fillna(0.0)

# --- evaluation harness (identical to NB2) -------------------------------------
def evaluate(cols, label):
    d = df.dropna(subset=["pair"]).copy()
    X = d[cols].to_numpy(dtype=float)
    y = d["y"].to_numpy(int)
    groups = d["pair"].to_numpy()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = None
    for name, mk in [("logistic", lambda: make_pipeline(
                          StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))),
                     ("hgb", lambda: HistGradientBoostingClassifier(
                          min_samples_leaf=10, random_state=SEED))]:
        oof = np.full(len(y), np.nan)
        Xi = np.nan_to_num(X) if name == "logistic" else X
        for tr, te in skf.split(Xi, y, groups):
            mdl = mk().fit(Xi[tr], y[tr])
            oof[te] = mdl.predict_proba(Xi[te])[:, 1]
        auc = roc_auc_score(y, oof)
        # pair-bootstrap CI
        pairs = d["pair"].unique()
        boot = []
        idx_by_pair = {p: np.where(groups == p)[0] for p in pairs}
        for _ in range(1000):
            samp = rng.choice(pairs, size=len(pairs), replace=True)
            ii = np.concatenate([idx_by_pair[p] for p in samp])
            try: boot.append(roc_auc_score(y[ii], oof[ii]))
            except ValueError: pass
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if best is None or auc > best["auc"]:
            best = {"label": label, "model": name, "auc": auc, "lo": lo, "hi": hi,
                    "n_feat": len(cols), "n_pairs": d["pair"].nunique()}
    return best

# --- specification ladder ------------------------------------------------------
specs = [("META_FLOOR (the 0.76 floor)",        META_FLOOR)]
specs += [("lexicon only",                      LEX)]
specs += [("META_FLOOR + lexicon",              META_FLOOR + LEX)]
if NEU:
    specs += [("neural only",                   NEU)]
    specs += [("META_FLOOR + neural",           META_FLOOR + NEU)]
    specs += [("META_FLOOR + lexicon + neural", META_FLOOR + LEX + NEU)]

results = [evaluate(cols, lab) for lab, cols in specs]
floor = results[0]["auc"]

print("\n" + "=" * 74)
print("SPECIFICATION LADDER  (best of logistic / HGB, delta vs the metadata floor)")
print("=" * 74)
print("%-34s %6s %18s %8s" % ("specification", "AUC", "95% CI", "Δ floor"))
for r in results:
    d = r["auc"] - floor
    dtxt = "  —  " if abs(d) < 1e-9 else "%+.3f" % d
    print("%-34s %6.3f  [%.3f-%.3f]  %7s  (%s, %d feat)"
          % (r["label"], r["auc"], r["lo"], r["hi"], dtxt, r["model"], r["n_feat"]))

pd.DataFrame(results).to_csv(OUT + "/nb3_specification_ladder.csv", index=False)
print("\nfloor AUC = %.3f  |  wrote nb3_specification_ladder.csv" % floor)
if not HAVE_NEURAL:
    print("NOTE: neural specs skipped — run score_bias_gpu.py on the GPU box, copy "
          "nb3_neural_features.csv back, and re-run this.")
