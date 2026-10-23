# --- Configuration -------------------------------------------------------------
# Set this to your own Wikimedia username: API policy asks for a contactable
# User-Agent, and PAWS runs are attributed to you.
WIKI_USERNAME = "MahimaSinghal"

LANG        = "en"      # Notebook 5 revisits this for Indic wikis
N_POSITIVES = 120       # articles to localise. 120 ~ 15 min. None = all ~2176 (~5 h).
N_CONTROLS  = 500       # matched controls to build
# A control matched only on total length and lifetime revision count can still be dormant
# during the window a detector would actually see. In a pilot, 45% of such controls had
# zero edits in the 90 days before their matched positive's onset, against 15% of
# positives - so a model could separate the classes on "did anything happen at all".
# Requiring window activity removes that shortcut.
CONTROL_REQUIRE_ACTIVITY = True
CONTROL_WINDOW_DAYS      = 90
RANDOM_SEED = 20261017
CACHE_DIR   = "./npov_cache"   # every API response is cached to disk, so re-runs are free
OUT_DIR     = "./npov_data"

import os, re, json, time, math, random, hashlib, calendar, datetime
import requests
import pandas as pd

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

API = "https://" + LANG + ".wikipedia.org/w/api.php"
UA  = ("npov-drift-research/0.1 (https://phabricator.wikimedia.org/T428896; "
       "User:" + WIKI_USERNAME + ") python-requests")
print("API:", API)
print("UA :", UA)

# --- Cached, rate-limited API client -------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
_last_call = [0.0]
MIN_INTERVAL = 0.10          # <=10 req/s, well inside API etiquette guidance
api_stats = {"calls": 0, "cache_hits": 0, "bytes": 0}

def api(**params):
    # One API query, memoised on disk by its parameters.
    params.setdefault("action", "query")
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    key = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        api_stats["cache_hits"] += 1
        with open(path) as f:
            return json.load(f)
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    r = SESSION.get(API, params=params, timeout=60)
    _last_call[0] = time.time()
    r.raise_for_status()
    api_stats["calls"] += 1
    api_stats["bytes"] += len(r.content)
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    with open(path, "w") as f:
        json.dump(data, f)
    return data

def api_all(listkey, **params):
    # Follow continuation and concatenate a list= result.
    out, cont = [], {}
    while True:
        d = api(**params, **cont)
        out += d.get("query", {}).get(listkey, [])
        if "continue" not in d:
            return out
        cont = d["continue"]

print(api(titles="Wikipedia:Neutral point of view", prop="info")["query"]["pages"][0]["title"],
      "- API reachable")

# Neutrality templates whose presence we treat as an NPOV dispute marker.
# Deliberately narrow: {{Weasel}}, {{Peacock}}, {{Advert}} and {{COI}} are *style* and
# *conflict-of-interest* tags; lumping them in is one way prior work inflates its positives.
NPOV_TAGS = [
    "POV", "NPOV", "Neutrality", "POV-check", "POV check", "Disputed-section",
    "POV section", "POV-section", "POV lead", "POV-lead", "POV statement", "POV-statement",
    "POV title", "POV-title", "Unbalanced", "Unbalanced section", "Undue weight",
    "Undue-weight", "Neutrality disputed",
]
NPOV_TAG_RE = re.compile(r"\{\{\s*(" +
                         "|".join(re.escape(t).replace("\\ ", "[ _]") for t in NPOV_TAGS) +
                         r")\s*(\||\}\})", re.IGNORECASE)

def revisions_between(title, start, end):
    # Revision stubs, oldest first, with timestamp in [start, end].
    out, cont = [], {}
    while True:
        d = api(prop="revisions", titles=title,
                rvprop="ids|timestamp|user|comment|size", rvlimit=500, rvdir="newer",
                rvstart=start.isoformat() + "T00:00:00Z",
                rvend=end.isoformat() + "T23:59:59Z", **cont)
        page = d["query"]["pages"][0]
        out += page.get("revisions", [])
        if "continue" not in d:
            return out
        cont = d["continue"]

def wikitext(revid):
    d = api(prop="revisions", revids=revid, rvprop="content", rvslots="main")
    pages = d.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None                               # deleted or suppressed revision
    slot = pages[0]["revisions"][0]["slots"]["main"]
    return None if slot.get("texthidden") else slot.get("content")

def npov_tags(txt):
    # [(normalised template name, date= value or None)] for every neutrality template present
    out = []
    for m in NPOV_TAG_RE.finditer(txt):
        tail = txt[m.start(): m.start() + 300]
        dm = re.search(r"date\s*=\s*(\w+\s+\d{4})", tail)
        out.append((re.sub(r"[ _-]+", " ", m.group(1)).strip().lower(),
                    dm.group(1).strip() if dm else None))
    return out

def tagged(revid, want_month=None):
    # With want_month set, ONLY a template whose date= matches that month counts.
    # This matters more than it looks: an article can carry an old, long-resolved {{POV}}
    # alongside the newly dated tag, and matching "any neutrality tag" then dates the
    # dispute to the wrong decade. That single mistake left-censored 32% of a pilot run.
    txt = wikitext(revid)
    if txt is None:
        return None
    tags = npov_tags(txt)
    if want_month is None:
        return bool(tags)
    want = want_month.strftime("%B %Y").lower()
    return any(d and d.strip().lower() == want for _, d in tags)

def _bisect_onset(revs, pred):
    # (outcome, index, fetches) for the first revision in revs satisfying pred.
    lo, hi, fetches = 0, len(revs) - 1, 1
    if not pred(revs[hi]["revid"]):
        return "none_at_end", lo, fetches
    fetches += 1
    if pred(revs[lo]["revid"]):
        return "at_start", lo, fetches
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        fetches += 1
        if pred(revs[mid]["revid"]):
            hi = mid
        else:
            lo = mid
    return "located", hi, fetches

def _month_offset(d, months):
    m = d.month - 1 + months
    return datetime.date(d.year + m // 12, m % 12 + 1, 1)

def find_onset(title, tag_month, slack_days=31, lookbacks=(1, 6, 24)):
    # First revision carrying the neutrality tag dated to this month.
    #
    # The lookback escalates. A tag dated "February 2025" is usually added in February,
    # but editors re-date tags and some tags are re-added after removal, so a one-month
    # lookback leaves a fifth of cases with the tag already present at the window start.
    # Widening is safe in strict mode, because we are hunting one specific dated template.
    eom = tag_month.replace(day=calendar.monthrange(tag_month.year, tag_month.month)[1])
    end = eom + datetime.timedelta(days=slack_days)
    rec = {"title": title, "tag_month": tag_month, "window_end": end,
           "fetches": 0, "status": None, "match_mode": None}
    revs, outcome = [], None

    for lb in lookbacks:
        start = _month_offset(tag_month, -lb)
        revs = revisions_between(title, start, end)
        rec.update({"window_start": start, "window_revs": len(revs), "lookback_months": lb})
        if not revs:
            continue
        outcome, idx, f = _bisect_onset(revs, lambda r: tagged(r, tag_month))
        rec["fetches"] += f
        rec["match_mode"] = "dated"
        if outcome == "located":
            return _fill_onset(rec, revs, idx)
        if outcome == "none_at_end":
            break              # the dated template is absent; try the lenient pass
        # "at_start": the tag predates this window, so widen and retry

    if not revs:
        rec["status"] = "no_revisions_in_window"
        return rec

    # lenient pass: the tag may have been added undated, with a bot filling in date= later
    outcome, idx, f = _bisect_onset(revs, lambda r: tagged(r, None))
    rec["fetches"] += f
    rec["match_mode"] = "any_tag"
    if outcome == "located":
        return _fill_onset(rec, revs, idx)
    rec["status"] = {"none_at_end": "untagged_at_window_end",
                     "at_start": "tagged_at_window_start"}[outcome]
    return rec

def _fill_onset(rec, revs, idx):
    # idx is the revision carrying the template dated to the category month. That is a
    # reliable anchor to the *right* dispute, but it is not always the human tagging:
    # a bot ("Dating maintenance tags: {{POV}}") adds date= to tags left undated, and
    # editors sometimes re-date. In a pilot run this hit 26% of rows. So when the
    # predecessor already carried a neutrality tag, walk back to the first revision in
    # the window that carries any tag, and treat *that* as the onset.
    anchor = revs[idx]
    rec["anchor_revid"] = anchor["revid"]
    rec["anchor_ts"] = anchor["timestamp"]
    rec["anchor_comment"] = anchor.get("comment") or ""
    onset, idx_o = anchor, idx
    rec["anchor_is_bot_dating"] = False
    rec["first_tag_censored"] = False

    prev_revid = revs[idx - 1]["revid"] if idx > 0 else None
    if prev_revid is not None and tagged(prev_revid, None):
        rec["anchor_is_bot_dating"] = True
        outcome_b, idx_b, f_b = _bisect_onset(revs[: idx + 1], lambda r: tagged(r, None))
        rec["fetches"] += f_b
        if outcome_b == "located":
            onset, idx_o = revs[idx_b], idx_b
        elif outcome_b == "at_start":
            # the tag was already there when the window opened: onset is left-censored,
            # so keep the anchor as a conservative upper bound and say so
            onset, idx_o = revs[0], 0
            rec["first_tag_censored"] = True

    rec["status"] = "located"
    rec["onset_revid"] = onset["revid"]
    rec["onset_ts"] = onset["timestamp"]
    rec["onset_user"] = onset.get("user")
    rec["onset_comment"] = onset.get("comment") or ""
    rec["onset_size"] = onset.get("size")
    rec["prev_revid"] = revs[idx_o - 1]["revid"] if idx_o > 0 else None
    rec["predecessor_clean"] = idx_o > 0 and not rec["first_tag_censored"]
    rec["anchor_lag_revs"] = idx - idx_o
    return rec


COMMENT_RE = re.compile(
    r"\b(?:npov|pov|neutral|neutrality|unbalanc|undue|one[- ]sided|biased?)\b", re.I)


BAD_CAT = re.compile(r"(stub|births|deaths|\d{4}|Wikipedia|Articles|Pages|CS1|Category:All)", re.I)

def topical_categories(title):
    d = api(prop="categories", titles=title, cllimit=500, clshow="!hidden")
    pages = d["query"]["pages"]
    cats = [c["title"] for c in pages[0].get("categories", [])] if pages else []
    return [c for c in cats if not BAD_CAT.search(c)]

def page_stats(title):
    d = api(prop="info|revisions", titles=title, rvprop="ids", rvlimit=500, rvdir="newer")
    p = d["query"]["pages"][0]
    if p.get("missing"):
        return None
    return {"title": title, "pageid": p["pageid"], "length": p.get("length", 0),
            "n_revs_capped": len(p.get("revisions", [])),
            "revs_truncated": "continue" in d}

def never_npov_tagged(title, max_comment_calls=4):
    # No neutrality tag in the current text, and none mentioned in revision summaries.
    cur = api(prop="revisions", titles=title, rvprop="content", rvslots="main", rvlimit=1)
    pages = cur.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return False
    txt = pages[0]["revisions"][0]["slots"]["main"].get("content") or ""
    if NPOV_TAG_RE.search(txt):
        return False
    cont, calls = {}, 0
    while calls < max_comment_calls:
        d = api(prop="revisions", titles=title, rvprop="comment", rvlimit=500,
                rvdir="newer", **cont)
        calls += 1
        for r in d["query"]["pages"][0].get("revisions", []):
            if COMMENT_RE.search(r.get("comment") or ""):
                return False
        if "continue" not in d:
            return True
        cont = d["continue"]
    return True        # screened as far as the call budget allows; flagged in the output

def active_in_window(title, end_ts, days=CONTROL_WINDOW_DAYS):
    """Did this article get edited at all in the days before end_ts?"""
    start = end_ts - datetime.timedelta(days=days)
    d = api(prop="revisions", titles=title, rvprop="ids", rvlimit=2, rvdir="newer",
            rvstart=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            rvend=end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
    pages = d.get("query", {}).get("pages", [])
    return bool(pages) and len(pages[0].get("revisions", [])) > 0

def match_control(pos_row, n_candidates=8):
    cats_here = topical_categories(pos_row.title)
    if not cats_here:
        return None
    target = (math.log1p(pos_row.length or 1), math.log1p(pos_row.n_revs_capped or 1))
    for cat in random.sample(cats_here, k=min(3, len(cats_here))):
        members = api_all("categorymembers", list="categorymembers", cmtitle=cat,
                          cmnamespace=0, cmlimit=200)
        random.shuffle(members)
        best = None
        # Screens are ordered cheapest-first: one call for page stats, one for window
        # activity, and only then the summary scan, which can cost five. Reordering these
        # cut the cost of building a control set by roughly half.
        for m in members[:n_candidates]:
            if m["title"] == pos_row.title:
                continue
            st = page_stats(m["title"])                                    # 1 call
            if not st or st["length"] < 2000:
                continue
            if CONTROL_REQUIRE_ACTIVITY and not active_in_window(          # 1 call
                    m["title"], pd.Timestamp(pos_row.onset_ts).to_pydatetime()):
                continue
            if not never_npov_tagged(m["title"]):                          # up to 5 calls
                continue
            d = math.hypot(math.log1p(st["length"]) - target[0],
                           math.log1p(st["n_revs_capped"]) - target[1])
            if best is None or d < best[0]:
                best = (d, st, cat)
            if best[0] < 0.25:            # close enough; stop paying for more candidates
                break
        if best:
            return {**best[1], "matched_to": pos_row.title, "match_category": best[2],
                    "match_distance": round(best[0], 3)}
    return None

# ---- stage: build controls for the full localised event set --------------------
located = pd.read_csv(os.path.join(OUT_DIR, "full_npov_events_en.csv"))
located = located[located.status == "located"].reset_index(drop=True)
print("loaded", len(located), "localised positives", flush=True)

ps = pd.DataFrame([s for s in (page_stats(t) for t in located.title) if s])
located = located.drop(columns=[c for c in ("length", "n_revs_capped", "pageid", "revs_truncated")
                                if c in located.columns]).merge(ps, on="title", how="left")
located = located.dropna(subset=["length", "onset_ts"])
targets = located.sample(frac=1.0, random_state=RANDOM_SEED).head(N_CONTROLS).reset_index(drop=True)
print("attempting controls for", len(targets), "positives", flush=True)

OUT = os.path.join(OUT_DIR, "full_npov_controls_en.csv")
controls, t0 = [], time.time()
for i, row in targets.iterrows():
    try:
        c = match_control(row)
    except Exception:
        c = None
    if c:
        controls.append(c)
    if (i + 1) % 10 == 0:
        el = time.time() - t0
        print("  %d/%d attempted, %d controls, %.0fs (%.1f s/attempt, %d calls)"
              % (i + 1, len(targets), len(controls), el, el / (i + 1), api_stats["calls"]), flush=True)
        pd.DataFrame(controls).to_csv(OUT, index=False)

ctl = pd.DataFrame(controls)
if len(ctl):
    ctl = ctl.drop_duplicates("pageid")
ctl.to_csv(OUT, index=False)
print("\nCONTROLS DONE:", len(ctl), "for", len(targets), "attempts", flush=True)
