#!/usr/bin/env python3
"""
Notebook 4, stage A'  —  back-dated text snapshots for the lead-time question.

CPU / API-bound. Read-only, cached, rate-limited. For each of the 421 both-active
pairs it snapshots BOTH the tagged article and its matched control at several lead
times before the tag: onset - {0, 30, 90, 180, 365} days (same calendar offset for the
control). Each snapshot is cleaned to sentences, exactly as Notebook 3 stage A.

The point: NB3 showed the neural subjectivity score adds real signal AT tagging time.
NB4 asks whether that textual bias is already present BEFORE a human tagged — i.e.
whether the signal has lead time, the actual early-warning question.

An article that did not exist yet at onset-lag has no revision and is recorded as
missing (revid null, no sentences); run_nb4.py reports attrition and a balanced panel.

Output (npov_data/):
  nb4_sentences.jsonl   one row per (article, lag): {title,y,pair,lag,revid,sentences}
"""
# --- Configuration -------------------------------------------------------------
WIKI_USERNAME = "MahimaSinghal"
LANG          = "en"
ROOT          = "."
EVENTS_CSV    = ROOT + "/npov_data/full_npov_events_en.csv"
CONTROLS_CSV  = ROOT + "/npov_data/full_npov_controls_en.csv"
NB2_FEATURES  = ROOT + "/npov_data/nb2_features.csv"      # to reuse the 421 both-active pairs
CACHE_DIR     = ROOT + "/npov_cache"
OUT_DIR       = ROOT + "/npov_data"
LAGS_DAYS     = [0, 30, 90, 180, 365]
MAX_CHARS     = 120_000
RANDOM_SEED   = 20261021

import os, re, json, time, hashlib, datetime
import requests, numpy as np, pandas as pd
try:
    import mwparserfromhell
    HAVE_MWP = True
except ImportError:
    HAVE_MWP = False
    print("WARNING: mwparserfromhell not found; using regex cleaner")

os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
API = "https://" + LANG + ".wikipedia.org/w/api.php"
UA  = ("npov-drift-research/0.4 (https://phabricator.wikimedia.org/T428896; "
       "User:" + WIKI_USERNAME + ") python-requests")
print("API:", API)

# --- Cached, rate-limited, RETRYING client (same as stage A) -------------------
SESSION = requests.Session(); SESSION.headers.update({"User-Agent": UA})
_last = [0.0]; MIN_INTERVAL = 0.12
stats = {"calls": 0, "hits": 0, "retries": 0}

def _cached_get(url, params, tag, max_retries=5):
    key  = hashlib.sha1((tag + url + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        stats["hits"] += 1
        with open(path) as f:
            return json.load(f)
    delay = 1.0
    for attempt in range(max_retries):
        w = MIN_INTERVAL - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        try:
            r = SESSION.get(url, params=params, timeout=60)
            _last[0] = time.time(); stats["calls"] += 1
            if r.status_code == 404:
                data = {"__http404": True}
                with open(path, "w") as f: json.dump(data, f)
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError("transient %d" % r.status_code)
            r.raise_for_status()
            data = r.json()
            with open(path, "w") as f: json.dump(data, f)
            return data
        except (requests.RequestException, ValueError):
            stats["retries"] += 1
            if attempt == max_retries - 1: raise
            time.sleep(delay); delay = min(delay * 2, 30)

def api(**params):
    params.setdefault("action", "query"); params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    d = _cached_get(API, params, "api")
    if "error" in d: raise RuntimeError(d["error"])
    return d

def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
def to_utc(s):
    try: return pd.to_datetime(s, utc=True, format="mixed")
    except (TypeError, ValueError): return pd.to_datetime(s, utc=True)
print("client ready")

# --- Pairs: reuse NB2's exact 421 both-active set ------------------------------
meta = pd.read_csv(NB2_FEATURES)
act  = meta.groupby("pair")["empty_window"].max()
both_active = set(act[act == 0].index)

ev = pd.read_csv(EVENTS_CSV); ev = ev[ev.status == "located"].copy()
ev["onset_ts"] = to_utc(ev.onset_ts)
ct = pd.read_csv(CONTROLS_CSV)
ct = ct.merge(ev[["title", "onset_ts"]].rename(columns={"title": "matched_to",
                                                        "onset_ts": "onset_ts"}),
              on="matched_to", how="inner")
# one record per article with its pair key and the pair's shared onset time
rows = []
for r in ev.itertuples(index=False):
    if r.title in both_active:
        rows.append({"title": r.title, "y": 1, "pair": r.title, "onset_ts": r.onset_ts})
for r in ct.itertuples(index=False):
    if r.matched_to in both_active:
        rows.append({"title": r.title, "y": 0, "pair": r.matched_to, "onset_ts": r.onset_ts})
arts = pd.DataFrame(rows)
arts["onset_ts"] = to_utc(arts.onset_ts)
print("%d pairs, %d articles x %d lags = %d snapshots"
      % (arts.pair.nunique(), len(arts), len(LAGS_DAYS), len(arts) * len(LAGS_DAYS)))

# --- text helpers (identical to stage A) ---------------------------------------
_REF=re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>",re.I|re.S); _TAG=re.compile(r"<[^>]+>")
_HEAD=re.compile(r"^\s*=+.*?=+\s*$",re.M); _FILE=re.compile(r"\[\[(?:File|Image):[^\]]*\]\]",re.I)
_SPACE=re.compile(r"[ \t]+"); _NL=re.compile(r"\n{2,}")
def to_prose(wt):
    if not wt: return ""
    txt=_REF.sub(" ",wt); txt=_FILE.sub(" ",txt)
    if HAVE_MWP:
        try: txt=mwparserfromhell.parse(txt).strip_code(normalize=True,collapse=True)
        except Exception: pass
    txt=_TAG.sub(" ",txt); txt=_HEAD.sub(" ",txt)
    txt=txt.replace("'''","").replace("''","")
    txt=re.sub(r"\{\{[^{}]*\}\}"," ",txt)
    txt=re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]",r"\2",txt)
    txt=re.sub(r"\[https?://\S+\s+([^\]]*)\]",r"\1",txt)
    txt=_SPACE.sub(" ",txt); txt=_NL.sub("\n",txt)
    return txt.strip()
_SENT=re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
def sentences(prose):
    out=[]
    for line in prose.split("\n"):
        line=line.strip()
        if len(line)<25: continue
        for s in _SENT.split(line):
            s=s.strip()
            if 25<=len(s)<=600 and sum(c.isalpha() for c in s)>=15: out.append(s)
    return out

def wikitext_asof(title, ts):
    d = api(prop="revisions", titles=title, rvprop="content|ids|timestamp",
            rvslots="main", rvlimit=1, rvdir="older", rvstart=iso(ts))
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"): return "", None
    revs = pages[0].get("revisions", [])
    if not revs: return "", None
    return (revs[0].get("slots", {}).get("main", {}).get("content") or ""), revs[0].get("revid")

# --- main loop -----------------------------------------------------------------
t0 = time.time(); out_rows = []; n = 0; total = len(arts) * len(LAGS_DAYS)
with open(os.path.join(OUT_DIR, "nb4_sentences.jsonl"), "w") as f:
    for r in arts.itertuples(index=False):
        for lag in LAGS_DAYS:
            n += 1
            ts = (r.onset_ts - pd.Timedelta(days=lag)).to_pydatetime()
            wt, revid = wikitext_asof(r.title, ts)
            sents = sentences(to_prose(wt[:MAX_CHARS])) if revid else []
            f.write(json.dumps({"title": r.title, "y": int(r.y), "pair": r.pair,
                                "lag": lag, "revid": revid, "sentences": sents}) + "\n")
            if n % 200 == 0 or n == total:
                print("  %5d/%d  %5.0fs  (live %d, cache %d, retries %d)"
                      % (n, total, time.time()-t0, stats["calls"], stats["hits"], stats["retries"]))

print("\nwrote nb4_sentences.jsonl")
