#!/usr/bin/env python3
"""
Notebook 3, stage A  —  text extraction + transparent lexicon features.

CPU / API-bound. No GPU. Runs anywhere with `requests`, `pandas`, `mwparserfromhell`.
Read-only on the public MediaWiki API, cached and rate-limited, no login, no edits.

What it does, for each of the 421 both-active pairs from Notebook 1/2:
  1. fetches the article's wikitext AS OF its window_end (the onset timestamp for a
     positive; the matched positive's onset for its control),
  2. strips wiki markup to plain prose and splits into sentences,
  3. computes length-normalised "loaded language" rates from published bias lexicons
     (Recasens 2013 bias lexicon, Hyland hedges, factive/assertive verbs, and
     Wikipedia's own WP:PEACOCK / WP:WEASEL word-to-watch lists).

This measures CONCURRENT detectability — does the text at tagging time separate a
tagged article from its matched untagged twin, over and above the 0.76 metadata floor.
Lead time (predicting BEFORE the human tagged) is a later notebook, not this one.

Outputs (in npov_data/):
  nb3_sentences.jsonl       one row per article: {title, y, pair, sentences:[...]}
                            — the input the GPU scorer (stage B) consumes.
  nb3_lexicon_features.csv  one row per article: the lexicon rates above.
"""

# --- Configuration -------------------------------------------------------------
WIKI_USERNAME = "MahimaSinghal"
LANG          = "en"
ROOT          = "."
EVENTS_CSV    = ROOT + "/npov_data/full_npov_events_en.csv"
CONTROLS_CSV  = ROOT + "/npov_data/full_npov_controls_en.csv"
CACHE_DIR     = ROOT + "/npov_cache"
OUT_DIR       = ROOT + "/npov_data"
WINDOW_DAYS   = 90        # snapshot is taken at window_end (onset); kept for provenance
RANDOM_SEED   = 20261021
MAX_CHARS     = 120_000   # guard against pathological huge pages before sentence split

import os, re, json, time, math, random, hashlib, datetime
import requests
import numpy as np
import pandas as pd

try:
    import mwparserfromhell
    HAVE_MWP = True
except ImportError:                                   # regex fallback keeps it runnable
    HAVE_MWP = False
    print("WARNING: mwparserfromhell not found; using the weaker regex cleaner")

os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

API = "https://" + LANG + ".wikipedia.org/w/api.php"
UA  = ("npov-drift-research/0.3 (https://phabricator.wikimedia.org/T428896; "
       "User:" + WIKI_USERNAME + ") python-requests")
print("API:", API)

# --- Cached, rate-limited, RETRYING client -------------------------------------
# Same lesson as Notebook 2: a transient failure must NEVER be cached as data.
# We cache only genuine 404s; 429/5xx/connection errors are retried with backoff.
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
        if w > 0:
            time.sleep(w)
        try:
            r = SESSION.get(url, params=params, timeout=60)
            _last[0] = time.time(); stats["calls"] += 1
            if r.status_code == 404:
                data = {"__http404": True}
                with open(path, "w") as f:
                    json.dump(data, f)
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError("transient %d" % r.status_code)
            r.raise_for_status()
            data = r.json()
            with open(path, "w") as f:                # only genuine successes are cached
                json.dump(data, f)
            return data
        except (requests.RequestException, ValueError) as e:
            stats["retries"] += 1
            if attempt == max_retries - 1:
                raise
            time.sleep(delay); delay = min(delay * 2, 30)

def api(**params):
    params.setdefault("action", "query")
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    d = _cached_get(API, params, "api")
    if "error" in d:
        raise RuntimeError(d["error"])
    return d

def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def to_utc(s):
    try:    return pd.to_datetime(s, utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, utc=True)

print("client ready")

# --- Load the exact same pairs Notebook 2 modelled -----------------------------
ev = pd.read_csv(EVENTS_CSV)
ev = ev[ev.status == "located"].copy()
ev["onset_ts"] = to_utc(ev.onset_ts)
ct = pd.read_csv(CONTROLS_CSV)
ct = ct.merge(ev[["title", "onset_ts"]].rename(columns={"title": "matched_to",
                                                        "onset_ts": "window_end_ts"}),
              on="matched_to", how="inner")
pairs = pd.concat([
    ev.assign(y=1, pair=ev.title, window_end_ts=ev.onset_ts)[["title","y","pair","window_end_ts"]],
    ct.assign(y=0, pair=ct.matched_to)[["title","y","pair","window_end_ts"]],
], ignore_index=True)
good = pairs.groupby("pair").y.agg(["sum", "count"])
good = good[(good["sum"] == 1) & (good["count"] == 2)].index
pairs = pairs[pairs.pair.isin(good)].reset_index(drop=True)
pairs["window_end_ts"] = to_utc(pairs.window_end_ts)
print("%d complete pairs, %d articles to fetch" % (len(good), len(pairs)))

# --- Fetch the article's wikitext as of a timestamp ----------------------------
def wikitext_asof(title, ts):
    """Most recent revision at or before ts, with its wikitext. '' if none/missing."""
    d = api(prop="revisions", titles=title, rvprop="content|ids|timestamp",
            rvslots="main", rvlimit=1, rvdir="older", rvstart=iso(ts))
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return "", None
    revs = pages[0].get("revisions", [])
    if not revs:
        return "", None
    slot = revs[0].get("slots", {}).get("main", {})
    return (slot.get("content") or ""), revs[0].get("revid")

# --- Wikitext -> plain prose ---------------------------------------------------
_REF   = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.I | re.S)
_TAG   = re.compile(r"<[^>]+>")
_HEAD  = re.compile(r"^\s*=+.*?=+\s*$", re.M)       # == Section headers ==
_FILE  = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.I)
_SPACE = re.compile(r"[ \t]+")
_NL    = re.compile(r"\n{2,}")

def to_prose(wikitext):
    if not wikitext:
        return ""
    txt = _REF.sub(" ", wikitext)
    txt = _FILE.sub(" ", txt)
    if HAVE_MWP:
        try:
            txt = mwparserfromhell.parse(txt).strip_code(normalize=True, collapse=True)
        except Exception:
            pass
    # residual markup cleanup (and the whole job in the regex-fallback case)
    txt = _TAG.sub(" ", txt)
    txt = _HEAD.sub(" ", txt)
    txt = txt.replace("'''", "").replace("''", "")
    txt = re.sub(r"\{\{[^{}]*\}\}", " ", txt)        # any leftover templates
    txt = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", txt)  # [[a|b]] -> b
    txt = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", txt)   # [url label] -> label
    txt = _SPACE.sub(" ", txt)
    txt = _NL.sub("\n", txt)
    return txt.strip()

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
def sentences(prose):
    out = []
    for line in prose.split("\n"):
        line = line.strip()
        if len(line) < 25:                           # skip list fragments, stubs, captions
            continue
        for s in _SENT.split(line):
            s = s.strip()
            if 25 <= len(s) <= 600 and sum(c.isalpha() for c in s) >= 15:
                out.append(s)
    return out

# --- Bias lexicons (published / documented provenance) -------------------------
# Wikipedia WP:PEACOCK words-to-watch (promotional puffery).
PEACOCK = set("""legendary acclaimed iconic visionary renowned celebrated eminent
prestigious remarkable extraordinary incredible unparalleled unmatched flawless
masterful genius brilliant seminal landmark definitive foremost leading world-class
award-winning cutting-edge state-of-the-art groundbreaking revolutionary""".split())
# WP:WEASEL vague-attribution phrases.
WEASEL = set("""arguably reportedly allegedly supposedly apparently seemingly
many some critics observers experts commentators sources others widely generally
commonly often reputedly""".split())
# Hyland (2005) hedges — epistemic uncertainty.
HEDGES = set("""may might could would should perhaps possibly probably likely
apparently seemingly presumably arguably suggest suggests indicate indicates
appear appears tend tends generally usually often sometimes relatively somewhat
fairly rather quite virtually essentially largely mostly partly""".split())
# Classic factive (Hooper 1975) and assertive verbs — stance-laden framing of claims.
FACTIVE   = set("""realize realized realizes know knew knows regret regretted
reveal revealed acknowledge acknowledged admit admitted recognize recognized
note noted notes understand understood""".split())
ASSERTIVE = set("""claim claimed claims insist insisted maintain maintained
assert asserted argue argued contend contended allege alleged purport purported
profess professed""".split())
# Recasens et al. (2013) bias lexicon — a documented subset of their highest-weight
# bias-inducing terms (subjective intensifiers + framing); provenance noted in Phab.
RECASENS = set("""controversial controversially so-called supposed notorious infamous
disputed questionable dubious propaganda regime terrorist freedom-fighter extremist
radical fundamentalist hardline staunch devout brutal ruthless heroic tragic
shocking stunning devastating massive overwhelming failed disastrous triumphant
successful powerful corrupt scandalous""".split())

LEXICONS = {"peacock": PEACOCK, "weasel": WEASEL, "hedge": HEDGES,
            "factive": FACTIVE, "assertive": ASSERTIVE, "bias_lex": RECASENS}
_WORD = re.compile(r"[a-z][a-z'-]+")

def lexicon_features(prose, sents):
    toks = _WORD.findall(prose.lower())
    ntok = max(len(toks), 1)
    bag  = set(toks)
    feats = {}
    for name, lex in LEXICONS.items():
        hits = sum(1 for t in toks if t in lex)
        feats["%s_rate" % name] = 1000.0 * hits / ntok      # hits per 1000 tokens
    # share of sentences carrying any subjective/bias-lexicon cue
    cue = PEACOCK | WEASEL | RECASENS
    flagged = 0
    for s in sents:
        st = set(_WORD.findall(s.lower()))
        if st & cue:
            flagged += 1
    feats["subjective_sent_share"] = flagged / max(len(sents), 1)
    feats["n_tokens"]    = ntok
    feats["n_sentences"] = len(sents)
    feats["mean_sent_len"] = (ntok / max(len(sents), 1))
    return feats

# --- Main loop -----------------------------------------------------------------
t0 = time.time()
sent_rows, feat_rows = [], []
for i, row in enumerate(pairs.itertuples(index=False), 1):
    wt, revid = wikitext_asof(row.title, row.window_end_ts.to_pydatetime())
    prose = to_prose(wt[:MAX_CHARS])
    sents = sentences(prose)
    feats = lexicon_features(prose, sents)
    feats.update({"title": row.title, "y": int(row.y), "pair": row.pair, "revid": revid})
    feat_rows.append(feats)
    sent_rows.append({"title": row.title, "y": int(row.y), "pair": row.pair,
                      "revid": revid, "sentences": sents})
    if i % 50 == 0 or i == len(pairs):
        el = time.time() - t0
        print("  %4d/%d  %5.0fs  (live %d, cache %d, retries %d)"
              % (i, len(pairs), el, stats["calls"], stats["hits"], stats["retries"]))

feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(os.path.join(OUT_DIR, "nb3_lexicon_features.csv"), index=False)
with open(os.path.join(OUT_DIR, "nb3_sentences.jsonl"), "w") as f:
    for r in sent_rows:
        f.write(json.dumps(r) + "\n")

empty = (feat_df.n_sentences == 0).sum()
print("\nwrote nb3_lexicon_features.csv  (%d rows, %d with no usable sentences)"
      % (len(feat_df), empty))
print("wrote nb3_sentences.jsonl       (feeds the GPU scorer)")
print("\nlexicon rate means (per 1000 tokens), control (0) vs tagged (1):")
cols = [c for c in feat_df.columns if c.endswith("_rate")] + ["subjective_sent_share"]
print(feat_df.groupby("y")[cols].mean().round(3).T.to_string())
