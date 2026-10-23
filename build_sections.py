#!/usr/bin/env python3
"""
Notebook 4b, stage A''  —  section-level extraction for the drift check.

NB4 found the WHOLE-ARTICLE subjectivity gap is flat over a year (no drift). This asks
whether drift hides at the SECTION level: does one section tilt, or does content pile
into one section (a volume proxy for due-weight), in the run-up to the tag?

Re-reads the SAME revisions NB4 used (all cache hits — no live API), splits each
snapshot into sections by MediaWiki headings, and emits two things:
  nb4_section_sizes.csv   per (article,lag): section-size concentration — computable
                          with NO GPU, analysed immediately by run_sections_size.py.
  nb4_sections.jsonl      per (article,lag): per-section sentence lists — the input to
                          the GPU subjectivity scorer (score_sections_gpu.py).

Lags {0,90,365} (tag / mid / far) are enough to show a trend against NB4's flat line.
"""
WIKI_USERNAME = "MahimaSinghal"; LANG = "en"
ROOT = "."
EVENTS_CSV = ROOT + "/npov_data/full_npov_events_en.csv"
CONTROLS_CSV = ROOT + "/npov_data/full_npov_controls_en.csv"
NB2_FEATURES = ROOT + "/npov_data/nb2_features.csv"
CACHE_DIR = ROOT + "/npov_cache"; OUT_DIR = ROOT + "/npov_data"
LAGS_DAYS = [0, 90, 365]; MAX_CHARS = 160_000

import os, re, json, time, hashlib
import requests, numpy as np, pandas as pd
try:
    import mwparserfromhell; HAVE_MWP = True
except ImportError:
    HAVE_MWP = False

API = "https://" + LANG + ".wikipedia.org/w/api.php"
UA  = ("npov-drift-research/0.5 (https://phabricator.wikimedia.org/T428896; "
       "User:" + WIKI_USERNAME + ") python-requests")
SESSION = requests.Session(); SESSION.headers.update({"User-Agent": UA})
_last = [0.0]; MIN_INTERVAL = 0.12; stats = {"calls": 0, "hits": 0, "retries": 0}

def _cached_get(url, params, tag, max_retries=5):
    key = hashlib.sha1((tag+url+json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        stats["hits"] += 1
        with open(path) as f: return json.load(f)
    delay = 1.0
    for attempt in range(max_retries):
        w = MIN_INTERVAL - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        try:
            r = SESSION.get(url, params=params, timeout=60)
            _last[0] = time.time(); stats["calls"] += 1
            if r.status_code == 404:
                d = {"__http404": True}
                with open(path, "w") as f: json.dump(d, f)
                return d
            if r.status_code in (429,500,502,503,504): raise requests.HTTPError("transient")
            r.raise_for_status(); d = r.json()
            with open(path, "w") as f: json.dump(d, f)
            return d
        except (requests.RequestException, ValueError):
            stats["retries"] += 1
            if attempt == max_retries-1: raise
            time.sleep(delay); delay = min(delay*2, 30)

def api(**p):
    p.setdefault("action","query"); p.setdefault("format","json"); p.setdefault("formatversion","2")
    d = _cached_get(API, p, "api")
    if "error" in d: raise RuntimeError(d["error"])
    return d
def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
def to_utc(s):
    try: return pd.to_datetime(s, utc=True, format="mixed")
    except (TypeError,ValueError): return pd.to_datetime(s, utc=True)

def wikitext_asof(title, ts):
    d = api(prop="revisions", titles=title, rvprop="content|ids", rvslots="main",
            rvlimit=1, rvdir="older", rvstart=iso(ts))
    pages = d.get("query",{}).get("pages",[])
    if not pages or pages[0].get("missing"): return "", None
    revs = pages[0].get("revisions",[])
    if not revs: return "", None
    return (revs[0].get("slots",{}).get("main",{}).get("content") or ""), revs[0].get("revid")

# --- section split + prose clean ----------------------------------------------
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
_REF=re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>",re.I|re.S); _TAG=re.compile(r"<[^>]+>")
_FILE=re.compile(r"\[\[(?:File|Image):[^\]]*\]\]",re.I); _SPACE=re.compile(r"[ \t]+"); _NL=re.compile(r"\n{2,}")
_SENT=re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
# sections that are not prose — exclude from due-weight (they inflate size/subjectivity noise)
_SKIP = re.compile(r"^(references|external links|further reading|see also|notes|"
                   r"bibliography|sources|citations|footnotes)$", re.I)

def clean(wt):
    if not wt: return ""
    t=_REF.sub(" ",wt); t=_FILE.sub(" ",t)
    if HAVE_MWP:
        try: t=mwparserfromhell.parse(t).strip_code(normalize=True, collapse=True)
        except Exception: pass
    t=_TAG.sub(" ",t); t=t.replace("'''","").replace("''","")
    t=re.sub(r"\{\{[^{}]*\}\}"," ",t); t=re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]",r"\2",t)
    t=re.sub(r"\[https?://\S+\s+([^\]]*)\]",r"\1",t); t=_SPACE.sub(" ",t); t=_NL.sub("\n",t)
    return t.strip()

def sents(prose):
    out=[]
    for line in prose.split("\n"):
        line=line.strip()
        if len(line)<25: continue
        for s in _SENT.split(line):
            s=s.strip()
            if 25<=len(s)<=600 and sum(c.isalpha() for c in s)>=15: out.append(s)
    return out

def split_sections(wt):
    """Return [(heading, raw_wikitext)] with 'lead' first; skips reference-type sections."""
    marks=[(m.start(), m.group(2)) for m in _HEADING.finditer(wt)]
    out=[]
    lead=wt[:marks[0][0]] if marks else wt
    out.append(("(lead)", lead))
    for i,(pos,head) in enumerate(marks):
        end = marks[i+1][0] if i+1 < len(marks) else len(wt)
        body = wt[pos:end]
        body = _HEADING.sub(" ", body, count=1)   # drop the heading line itself
        if not _SKIP.match(head.strip()):
            out.append((head.strip(), body))
    return out

def hhi(sizes):
    tot=sum(sizes)
    if tot<=0: return np.nan
    p=np.array(sizes)/tot
    return float((p*p).sum())                      # 1=all in one section, ->0 many equal

def gini(sizes):
    a=np.sort(np.array([s for s in sizes if s>=0], dtype=float))
    n=len(a)
    if n==0 or a.sum()==0: return np.nan
    return float((2*np.arange(1,n+1)-n-1).dot(a)/(n*a.sum()))

# --- pairs: both-active 421 ----------------------------------------------------
meta=pd.read_csv(NB2_FEATURES); act=meta.groupby("pair")["empty_window"].max()
both=set(act[act==0].index)
ev=pd.read_csv(EVENTS_CSV); ev=ev[ev.status=="located"].copy(); ev["onset_ts"]=to_utc(ev.onset_ts)
ct=pd.read_csv(CONTROLS_CSV).merge(ev[["title","onset_ts"]].rename(columns={"title":"matched_to"}),
                                   on="matched_to", how="inner")
rows=[]
for r in ev.itertuples(index=False):
    if r.title in both: rows.append({"title":r.title,"y":1,"pair":r.title,"onset_ts":r.onset_ts})
for r in ct.itertuples(index=False):
    if r.matched_to in both: rows.append({"title":r.title,"y":0,"pair":r.matched_to,"onset_ts":r.onset_ts})
arts=pd.DataFrame(rows); arts["onset_ts"]=to_utc(arts.onset_ts)
print("%d pairs, %d articles x %d lags" % (arts.pair.nunique(), len(arts), len(LAGS_DAYS)))

# --- main loop -----------------------------------------------------------------
t0=time.time(); size_rows=[]; n=0; total=len(arts)*len(LAGS_DAYS)
with open(os.path.join(OUT_DIR,"nb4_sections.jsonl"),"w") as fj:
    for r in arts.itertuples(index=False):
        for lag in LAGS_DAYS:
            n+=1
            ts=(r.onset_ts - pd.Timedelta(days=lag)).to_pydatetime()
            wt, revid = wikitext_asof(r.title, ts)
            secs=[]
            if revid:
                for head, body in split_sections(wt[:MAX_CHARS]):
                    ss=sents(clean(body))
                    secs.append({"heading":head, "nbytes":len(body), "sentences":ss})
            fj.write(json.dumps({"title":r.title,"y":int(r.y),"pair":r.pair,"lag":lag,
                                 "revid":revid,"sections":secs})+"\n")
            prose_secs=[s for s in secs if s["sentences"]]
            sizes=[s["nbytes"] for s in prose_secs]
            size_rows.append({"title":r.title,"y":int(r.y),"pair":r.pair,"lag":lag,"revid":revid,
                              "n_sections":len(prose_secs),
                              "size_hhi":hhi(sizes) if sizes else np.nan,
                              "size_gini":gini(sizes) if sizes else np.nan,
                              "max_sec_frac":(max(sizes)/sum(sizes)) if sizes and sum(sizes)>0 else np.nan})
            if n%300==0 or n==total:
                print("  %5d/%d  %5.0fs (live %d, cache %d)" % (n,total,time.time()-t0,stats["calls"],stats["hits"]))

pd.DataFrame(size_rows).to_csv(os.path.join(OUT_DIR,"nb4_section_sizes.csv"), index=False)
print("\nwrote nb4_section_sizes.csv (no-GPU due-weight-by-volume signal)")
print("wrote nb4_sections.jsonl   (feeds score_sections_gpu.py)")
