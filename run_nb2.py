# --- Configuration -------------------------------------------------------------
WIKI_USERNAME = "MahimaSinghal"
LANG          = "en"

EVENTS_CSV   = "./npov_data/full_npov_events_en.csv"      # from Notebook 1
CONTROLS_CSV = "./npov_data/full_npov_controls_en.csv"
WINDOW_DAYS  = 90
NEW_ACCT_DAYS = 30        # "new account" = registered within this many days of the window end
RANDOM_SEED  = 20261017
CACHE_DIR    = "./npov_cache"
OUT_DIR      = "./npov_data"
USE_PAGEVIEWS = True      # one extra REST call per article; set False to skip

import os, re, json, time, math, random, hashlib, datetime, urllib.parse
from collections import Counter
import requests
import numpy as np
import pandas as pd

os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

API = "https://" + LANG + ".wikipedia.org/w/api.php"
UA  = ("npov-drift-research/0.2 (https://phabricator.wikimedia.org/T428896; "
       "User:" + WIKI_USERNAME + ") python-requests")
print("API:", API)

# --- Cached, rate-limited clients ----------------------------------------------
SESSION = requests.Session(); SESSION.headers.update({"User-Agent": UA})
_last = [0.0]; MIN_INTERVAL = 0.12
stats = {"calls": 0, "hits": 0}

def _cached_get(url, params, tag):
    key = hashlib.sha1((tag + url + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        stats["hits"] += 1
        with open(path) as f:
            return json.load(f)
    w = MIN_INTERVAL - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    r = SESSION.get(url, params=params, timeout=60)
    _last[0] = time.time(); stats["calls"] += 1
    if r.status_code == 404:
        data = {"__http404": True}
    else:
        r.raise_for_status()
        data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    return data

def api(**params):
    params.setdefault("action", "query")
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    d = _cached_get(API, params, "api")
    if "error" in d:
        raise RuntimeError(d["error"])
    return d

def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def to_utc(s):
    """Parse timestamps that may mix 'Z' and '+00:00' forms, across pandas versions."""
    try:
        return pd.to_datetime(s, utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, utc=True)

print("client ready")

# --- Load Notebook 1's output --------------------------------------------------
ev = pd.read_csv(EVENTS_CSV)
ev = ev[ev.status == "located"].copy()
ev["onset_ts"] = to_utc(ev.onset_ts)
ct = pd.read_csv(CONTROLS_CSV)

# each control inherits the calendar window of the positive it was matched to
ct = ct.merge(ev[["title", "onset_ts"]].rename(columns={"title": "matched_to",
                                                        "onset_ts": "window_end_ts"}),
              on="matched_to", how="inner")

pairs = pd.concat([
    ev.assign(y=1, pair=ev.title, window_end_ts=ev.onset_ts)[
        ["title", "y", "pair", "window_end_ts"]],
    ct.assign(y=0, pair=ct.matched_to)[["title", "y", "pair", "window_end_ts"]],
], ignore_index=True)

# keep only complete pairs, so the design stays balanced
good = pairs.groupby("pair").y.agg(["sum", "count"])
good = good[(good["sum"] == 1) & (good["count"] == 2)].index
pairs = pairs[pairs.pair.isin(good)].reset_index(drop=True)

print("%d located positives, %d controls loaded, %d complete pairs kept"
      % (len(ev), len(ct), len(good)))
print(pairs.y.value_counts().to_string())

REVERTING_TAGS = {"mw-undo", "mw-rollback", "mw-manual-revert"}

def revisions_in(title, start, end):
    """Revisions with start <= ts < end, oldest first."""
    out, cont = [], {}
    while True:
        d = api(prop="revisions", titles=title,
                rvprop="ids|timestamp|user|comment|size|tags|flags",
                rvlimit=500, rvdir="newer", rvstart=iso(start), rvend=iso(end), **cont)
        pages = d.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return []
        out += pages[0].get("revisions", [])
        if "continue" not in d:
            break
        cont = d["continue"]
    cut = iso(end)
    return [r for r in out if r["timestamp"] < cut]

_reg_cache = {}
def registrations(users):
    """Account registration timestamps, batched 50 at a time and memoised."""
    want = sorted({u for u in users if u and not re.match(r"^\d+\.|:", u) and u not in _reg_cache})
    for i in range(0, len(want), 50):
        chunk = want[i:i + 50]
        d = api(list="users", ususers="|".join(chunk), usprop="registration")
        for u in d.get("query", {}).get("users", []):
            _reg_cache[u.get("name")] = u.get("registration")
    return {u: _reg_cache.get(u) for u in users}

def pageviews_sum(title, start, end):
    if not USE_PAGEVIEWS:
        return np.nan
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           + LANG + ".wikipedia/all-access/user/"
           + urllib.parse.quote(title.replace(" ", "_"), safe="")
           + "/daily/" + start.strftime("%Y%m%d") + "/" + end.strftime("%Y%m%d"))
    try:
        d = _cached_get(url, {}, "pv")
    except Exception:
        return np.nan
    if d.get("__http404") or "items" not in d:
        return np.nan
    return float(sum(i.get("views", 0) for i in d["items"]))

def features(title, end_ts, days=WINDOW_DAYS):
    start = end_ts - datetime.timedelta(days=days)
    revs = revisions_in(title, start, end_ts)
    talk = revisions_in("Talk:" + title, start, end_ts)
    f = {"n_revs": float(len(revs)), "talk_revs": float(len(talk)),
         "empty_window": 1.0 if not revs else 0.0,
         "pageviews": pageviews_sum(title, start, end_ts)}
    if not revs:
        return f
    users = [r.get("user") or "?" for r in revs]
    counts = Counter(users)
    n = float(len(revs))
    shares = np.array([c / n for c in counts.values()])
    sizes = [float(r.get("size") or 0) for r in revs]
    deltas = np.diff(sizes) if len(sizes) > 1 else np.array([0.0])
    tagsets = [set(r.get("tags") or []) for r in revs]
    reg = registrations(users)
    cutoff = end_ts - datetime.timedelta(days=NEW_ACCT_DAYS)

    def is_new(u):
        t = reg.get(u)
        if not t:
            return False
        try:
            return pd.to_datetime(t, utc=True) >= cutoff
        except Exception:
            return False

    f.update({
        "n_editors": float(len(counts)),
        "editor_hhi": float((shares ** 2).sum()),
        "top_editor_share": float(shares.max()),
        "anon_share": float(np.mean([1.0 if r.get("anon") else 0.0 for r in revs])),
        "bot_share": float(np.mean([1.0 if re.search(r"bot\b", u, re.I) else 0.0 for u in users])),
        "minor_share": float(np.mean([1.0 if r.get("minor") else 0.0 for r in revs])),
        "reverting_share": float(np.mean([1.0 if (t & REVERTING_TAGS) else 0.0 for t in tagsets])),
        "reverted_share": float(np.mean([1.0 if "mw-reverted" in t else 0.0 for t in tagsets])),
        "new_account_edit_share": float(np.mean([1.0 if is_new(u) else 0.0 for u in users])),
        "revs_per_day": float(n / days),
        "max_revs_one_day": float(max(Counter(r["timestamp"][:10] for r in revs).values())),
        "size_end": sizes[-1],
        "size_net_delta": sizes[-1] - sizes[0],
        "size_abs_churn": float(np.abs(deltas).sum()),
        "size_delta_std": float(np.std(deltas)),
        "empty_summary_share": float(np.mean([1.0 if not (r.get("comment") or "").strip() else 0.0
                                              for r in revs])),
    })
    return f

print("feature functions defined")

# --- Build the feature matrix --------------------------------------------------
rows, t0 = [], time.time()
for i, r in pairs.iterrows():
    try:
        f = features(r.title, r.window_end_ts.to_pydatetime())
    except Exception as e:
        f = {"error": type(e).__name__}
    rows.append({"title": r.title, "y": r.y, "pair": r.pair,
                 "window_end_ts": r.window_end_ts, **f})
    if (i + 1) % 20 == 0 or i + 1 == len(pairs):
        el = time.time() - t0
        print("  %d/%d  %5.0fs  (%.1f s/article, %d live calls)"
              % (i + 1, len(pairs), el, el / (i + 1), stats["calls"]))

X = pd.DataFrame(rows)
X.to_csv(os.path.join(OUT_DIR, "nb2_features.csv"), index=False)
print("\nfeature matrix:", X.shape)
print("rows with an error:", X.get("error", pd.Series(dtype=object)).notna().sum())

FEATURES = ["n_revs", "revs_per_day", "max_revs_one_day", "n_editors", "editor_hhi",
            "top_editor_share", "anon_share", "bot_share", "minor_share",
            "reverting_share", "reverted_share", "new_account_edit_share",
            "size_end", "size_net_delta", "size_abs_churn", "size_delta_std",
            "empty_summary_share", "talk_revs", "pageviews", "empty_window"]
FEATURES = [f for f in FEATURES if f in X.columns]

D = X.dropna(subset=["y"]).copy()
for c in FEATURES:
    D[c] = pd.to_numeric(D[c], errors="coerce")
# median-impute, and keep an explicit missingness flag for pageviews
if "pageviews" in FEATURES:
    D["pageviews_missing"] = D.pageviews.isna().astype(float)
    FEATURES = FEATURES + ["pageviews_missing"]
D[FEATURES] = D[FEATURES].fillna(D[FEATURES].median())

# drop pairs that lost a side
cnt = D.groupby("pair").y.agg(["sum", "count"])
keep = cnt[(cnt["sum"] == 1) & (cnt["count"] == 2)].index
D = D[D.pair.isin(keep)].reset_index(drop=True)
print("modelling set: %d rows, %d pairs, %d features" % (len(D), D.pair.nunique(), len(FEATURES)))
print("\nclass means (raw, unstandardised):")
print(D.groupby("y")[FEATURES].mean().T.to_string(float_format=lambda v: "%.3f" % v))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

def cv_scores(model, D, features, n_splits=5, seed=RANDOM_SEED):
    y = D.y.values.astype(int)
    g = D.pair.values
    Xm = D[features].values
    oof = np.full(len(D), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_auc = []
    for tr, te in cv.split(Xm, y, groups=g):
        m = model()
        m.fit(Xm[tr], y[tr])
        p = m.predict_proba(Xm[te])[:, 1]
        oof[te] = p
        if len(set(y[te])) == 2:
            fold_auc.append(roc_auc_score(y[te], p))
    return oof, np.array(fold_auc)

def boot_ci(y, p, n=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s])) == 2:
            out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

MODELS = {
    "logistic": lambda: make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0)),
    # min_samples_leaf below sklearn's default of 20: with a few hundred pairs the
    # default makes every split illegal and the model predicts a constant.
    "grad boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.06, min_samples_leaf=10,
        random_state=RANDOM_SEED),
}

results = {}
yv = D.y.values.astype(int)
for name, ctor in MODELS.items():
    oof, folds = cv_scores(ctor, D, FEATURES)
    auc = roc_auc_score(yv, oof)
    lo, hi = boot_ci(yv, oof)
    ap = average_precision_score(yv, oof)
    results[name] = {"auc": auc, "ci": (lo, hi), "folds": folds, "ap": ap, "oof": oof}
    print("%-14s ROC-AUC %.3f  [95%% CI %.3f-%.3f]   per-fold %s   PR-AUC %.3f"
          % (name, auc, lo, hi, np.round(folds, 3), ap))

active_pairs = (D.assign(act=(D.n_revs > 0).astype(int))
                 .groupby("pair").act.min())
keep_act = set(active_pairs[active_pairs == 1].index)
Da = D[D.pair.isin(keep_act)].reset_index(drop=True)
print("pairs where both sides were edited in the window: %d of %d"
      % (Da.pair.nunique(), D.pair.nunique()))

if Da.pair.nunique() >= 20:
    ya = Da.y.values.astype(int)
    for name, ctor in MODELS.items():
        oof_a, folds_a = cv_scores(ctor, Da, FEATURES,
                                   n_splits=min(5, Da.pair.nunique() // 4))
        lo, hi = boot_ci(ya, oof_a)
        print("  %-14s ROC-AUC %.3f  [95%% CI %.3f-%.3f]  (both-active subset)"
              % (name, roc_auc_score(ya, oof_a), lo, hi))
else:
    print("too few both-active pairs to recompute; skipping")

# How much of that is one feature? Univariate AUC, direction-corrected.
uni = []
for f in FEATURES:
    v = D[f].values
    if np.nanstd(v) == 0:
        continue
    a = roc_auc_score(yv, v)
    uni.append({"feature": f, "auc": max(a, 1 - a),
                "direction": "higher in tagged" if a > 0.5 else "higher in control"})
uni = pd.DataFrame(uni).sort_values("auc", ascending=False).reset_index(drop=True)
print("Single-feature AUC (what one metadata column achieves alone):\n")
print(uni.to_string(index=False, float_format=lambda v: "%.3f" % v))

SOCIAL = ["editor_hhi", "top_editor_share", "anon_share", "bot_share", "minor_share",
          "reverting_share", "reverted_share", "new_account_edit_share", "empty_summary_share"]
NO_ARTIFACT = [f for f in FEATURES if f not in ("pageviews", "pageviews_missing", "empty_window")]
SOCIAL = [f for f in SOCIAL if f in FEATURES]

act = D.assign(_a=(D.n_revs > 0).astype(int)).groupby("pair")._a.min()
Dact = D[D.pair.isin(set(act[act == 1].index))]

def spec(df, feats, label):
    if df.pair.nunique() < 20 or not feats:
        print("  %-50s skipped (too few pairs)" % label)
        return None
    row = {"spec": label, "n_pairs": int(df.pair.nunique())}
    for name, ctor in MODELS.items():
        oof, _ = cv_scores(ctor, df, feats, n_splits=min(5, df.pair.nunique() // 4))
        row[name] = round(roc_auc_score(df.y.values.astype(int), oof), 3)
    print("  %-50s pairs=%3d  %s" % (label, row["n_pairs"],
          "  ".join("%s %.3f" % (k, row[k]) for k in MODELS)))
    return row

print("NESTED SPECIFICATIONS\n")
ladder = [r for r in [
    spec(D,    FEATURES,    "A. all features, all pairs (upper bound)"),
    spec(Dact, FEATURES,    "B. all features, both-active pairs"),
    spec(Dact, NO_ARTIFACT, "C. no traffic, no empty_window  <-- THE FLOOR"),
    spec(Dact, SOCIAL,      "D. social features only (no activity volume)"),
    spec(Dact, ["n_revs"],  "E. revision count alone (dumbest baseline)"),
    spec(Dact, ["talk_revs"], "F. talk-page revisions alone"),
] if r]
pd.DataFrame(ladder).to_csv(os.path.join(OUT_DIR, "nb2_specification_ladder.csv"), index=False)

floor = max(v for k, v in (ladder[2] if len(ladder) > 2 else {}).items()
            if k in MODELS) if len(ladder) > 2 else float("nan")
print("\nFLOOR (spec C, best model): %.3f" % floor)
print("Any content signal must be reported as a delta over THIS number, on these pairs.")

# Precision@k: the deployable framing. A reviewer opens the top k% of the queue.
best = max(results, key=lambda k: results[k]["auc"])
p = results[best]["oof"]
print("ranking by the %s model's out-of-fold score\n" % best)
print("  k        precision   (base rate %.2f)" % yv.mean())
for frac in (0.05, 0.10, 0.20, 0.50):
    k = max(1, int(round(frac * len(p))))
    top = np.argsort(-p)[:k]
    print("  top %3.0f%%   %.3f" % (100 * frac, yv[top].mean()))

# Temporal holdout: train on older disputes, test on the most recent fifth.
Dt = D.copy()
Dt["t"] = to_utc(Dt.window_end_ts)
pair_t = Dt.groupby("pair").t.min().sort_values()
cut = pair_t.quantile(0.80)
tr_pairs = set(pair_t[pair_t <= cut].index); te_pairs = set(pair_t[pair_t > cut].index)
tr = Dt.pair.isin(tr_pairs).values; te = Dt.pair.isin(te_pairs).values

print("train pairs %d (to %s), test pairs %d (after)"
      % (len(tr_pairs), cut.date(), len(te_pairs)))
for name, ctor in MODELS.items():
    m = ctor(); m.fit(Dt.loc[tr, FEATURES].values, Dt.loc[tr, "y"].values.astype(int))
    pt = m.predict_proba(Dt.loc[te, FEATURES].values)[:, 1]
    yt = Dt.loc[te, "y"].values.astype(int)
    if len(set(yt)) == 2:
        print("  %-14s future-year ROC-AUC %.3f" % (name, roc_auc_score(yt, pt)))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for name in results:
    fpr, tpr, _ = roc_curve(yv, results[name]["oof"])
    ax.plot(fpr, tpr, label="%s (AUC %.3f)" % (name, results[name]["auc"]))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Metadata-only floor"); ax.legend(frameon=False, fontsize=8)

ax = axes[1]
top = uni.head(10).iloc[::-1]
ax.barh(top.feature, top.auc, color="#3366cc")
ax.axvline(0.5, color="k", lw=0.8, ls="--")
ax.set_xlim(0.45, max(0.75, top.auc.max() + 0.03))
ax.set_xlabel("single-feature AUC"); ax.set_title("What one column achieves alone")
for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig("./npov_data/nb2_figure.png", dpi=140)

# --- Save ---------------------------------------------------------------------
summary = {
    "task": "https://phabricator.wikimedia.org/T428896",
    "notebook": "02_null_model",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "wiki": LANG + ".wikipedia.org",
    "window_days": WINDOW_DAYS,
    "n_pairs": int(D.pair.nunique()),
    "n_rows": int(len(D)),
    "n_features": len(FEATURES),
    "features": FEATURES,
    "models": {k: {"roc_auc": round(v["auc"], 4),
                   "ci95": [round(v["ci"][0], 4), round(v["ci"][1], 4)],
                   "pr_auc": round(v["ap"], 4),
                   "fold_aucs": [round(float(x), 4) for x in v["folds"]]}
               for k, v in results.items()},
    "top_univariate": uni.head(8).to_dict("records"),
    "live_api_calls": stats["calls"],
}
with open(os.path.join(OUT_DIR, "nb2_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary["models"], indent=2))

best_auc = max(v["auc"] for v in results.values())
verdict = ("metadata DOMINATES - re-scope to lead time + cross-lingual transfer" if best_auc >= 0.80
           else "PARTIAL - content signals must be reported as a delta over this floor"
           if best_auc > 0.65 else
           "metadata is WEAK - content signals have real room")
print("\nbest metadata-only AUC: %.3f" % best_auc)
print("pre-registered reading:", verdict)