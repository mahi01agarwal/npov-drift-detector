# --- Configuration -------------------------------------------------------------
# Set this to your own Wikimedia username: API policy asks for a contactable
# User-Agent, and PAWS runs are attributed to you.
WIKI_USERNAME = "MahimaSinghal"

LANG        = "en"      # Notebook 5 revisits this for Indic wikis
N_POSITIVES = None       # articles to localise. 120 ~ 15 min. None = all ~2176 (~5 h).
N_CONTROLS  = 120       # matched controls to build
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
MIN_INTERVAL = 0.15          # <=10 req/s, well inside API etiquette guidance
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

MONTHS = {m: i for i, m in enumerate(calendar.month_name) if m}
DATED_PREFIX = "Wikipedia neutral point of view disputes from"

cats = api_all("allcategories", list="allcategories", acprefix=DATED_PREFIX,
               aclimit=500, acprop="size")

def cat_month(name):
    m = re.search(r"from (\w+) (\d{4})$", name)
    return datetime.date(int(m.group(2)), MONTHS[m.group(1)], 1) if m else None

def cat_name(c):
    return c.get("category") or c.get("*")      # formatversion 2 vs 1

pool = pd.DataFrame([{"category": "Category:" + cat_name(c),
                      "month": cat_month(cat_name(c)),
                      "n_pages": c.get("pages", 0)} for c in cats])
pool = pool.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

print(len(pool), "dated categories,", pool.n_pages.sum(), "tagged pages,",
      pool.month.min().strftime("%Y-%m"), "->", pool.month.max().strftime("%Y-%m"))
by_year = pool.assign(year=pool.month.map(lambda d: d.year)).groupby("year").n_pages.sum()
print("\nOpen disputes by year the tag was placed:")
print(by_year.to_string())

# Members of every dated category -> the positive pool.
rows = []
for _, c in pool.iterrows():
    for m in api_all("categorymembers", list="categorymembers", cmtitle=c.category,
                     cmnamespace=0, cmlimit=500):
        rows.append({"title": m["title"], "pageid": m["pageid"], "tag_month": c.month})

positives = pd.DataFrame(rows).drop_duplicates("pageid").reset_index(drop=True)
print(len(positives), "distinct tagged articles")
print("API calls so far:", api_stats["calls"], "(cache hits", api_stats["cache_hits"], ")")

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

# smoke test on a case checked by hand
print(json.dumps(find_onset("Jews as the chosen people", datetime.date(2025, 5, 1)),
                 indent=1, default=str))

# --- Localise a sample --------------------------------------------------------
sample = positives.sample(frac=1.0, random_state=RANDOM_SEED)
if N_POSITIVES:
    sample = sample.head(N_POSITIVES)
sample = sample.reset_index(drop=True)

records, t0 = [], time.time()
for i, row in sample.iterrows():
    try:
        records.append(find_onset(row.title, row.tag_month))
    except Exception as e:
        records.append({"title": row.title, "tag_month": row.tag_month,
                        "status": "error: " + type(e).__name__ + ": " + str(e)})
    if (i + 1) % 10 == 0 or i + 1 == len(sample):
        el = time.time() - t0
        print("  %d/%d  %5.0fs elapsed  (%.1f s/article, %d live calls)"
              % (i + 1, len(sample), el, el / (i + 1), api_stats["calls"]))

events = pd.DataFrame(records)
print("\nOutcome of localisation:")
print(events.status.value_counts().to_string())

located = events[events.status == "located"].copy()
located["onset_ts"] = pd.to_datetime(located.onset_ts)

print("localised %d/%d = %.1f%%" % (len(located), len(events), 100 * len(located) / len(events)))
print("content fetches per article: median %.0f, max %.0f"
      % (located.fetches.median(), located.fetches.max()))
print("revisions in search window:  median %.0f, max %.0f"
      % (located.window_revs.median(), located.window_revs.max()))
print("\nhow the tag was matched (dated = exact template for this month):")
print(located.match_mode.value_counts().to_string())
print("\nlookback needed to find it (months before the stated month):")
print(located.lookback_months.value_counts().sort_index().to_string())

bot = located.anchor_is_bot_dating.fillna(False).astype(bool)
print("\nrows where the dated template was placed by a bot or re-dating edit, so the")
print("onset was walked back to the human tagging: %d (%.1f%%)" % (bot.sum(), 100 * bot.mean()))
print("  revisions walked back, median: %.0f" % located.loc[bot, "anchor_lag_revs"].median()
      if bot.any() else "  (none)")
cens = located.first_tag_censored.fillna(False).astype(bool)
print("rows still left-censored after the walk-back (onset predates the window): %d" % cens.sum())

# Does the located revision land inside the month the template claims? A strong sanity check.
in_month = located.apply(
    lambda r: (r.onset_ts.year, r.onset_ts.month) == (r.tag_month.year, r.tag_month.month), axis=1)
print("\nonset revision falls in the month stated by the template: %.1f%%" % (100 * in_month.mean()))
print("(mismatches are expected: a tag added on the 31st can be dated to the next month, and "
      "bots re-date undated tags)")

SECTION_TAGS = re.compile(r"^(POV[ _-]?section|POV[ _-]?lead|POV[ _-]?statement|"
                          r"Disputed-section|Unbalanced[ _]section)$", re.I)

def enrich_scope(revid, tag_month=None):
    # Which template fired, where in the page, and under which heading.
    txt = wikitext(revid)
    if txt is None:
        return {}
    want = tag_month.strftime("%B %Y").lower() if tag_month is not None else None
    chosen = None
    for m in NPOV_TAG_RE.finditer(txt):
        tail = txt[m.start(): m.start() + 300]
        dm = re.search(r"date\s*=\s*(\w+\s+\d{4})", tail)
        d = dm.group(1).strip() if dm else None
        if chosen is None:
            chosen = (m, d)
        if want and d and d.lower() == want:      # prefer the tag for *this* dispute
            chosen = (m, d)
            break
    if chosen is None:
        return {}
    m, tdate = chosen
    name = re.sub(r"[ _-]+", " ", m.group(1)).strip().lower()
    before = txt[:m.start()]
    headings = re.findall(r"^(={2,})\s*(.+?)\s*\1\s*$", before, re.M)
    is_section_template = bool(SECTION_TAGS.match(name.replace(" ", "")))
    scope = "section" if (is_section_template or headings) else "article"
    return {"template": name, "scope": scope,
            "section": headings[-1][1] if headings else None,
            "template_date": tdate,
            "article_chars": len(txt)}

ENRICH_COLS = ["template", "scope", "section", "template_date", "article_chars"]
enr = pd.DataFrame([enrich_scope(int(r.onset_revid), r.tag_month)
                    for _, r in located.iterrows()]).reindex(columns=ENRICH_COLS)
located = pd.concat([located.reset_index(drop=True), enr], axis=1)

print("Template used:")
print(located.template.value_counts().to_string())
print("\nScope:")
print(located.scope.value_counts(normalize=True).to_string())
print("\nArticle size at tagging: median {:,.0f} chars".format(located.article_chars.median()))


COMMENT_RE = re.compile(
    r"\b(?:npov|pov|neutral|neutrality|unbalanc|undue|one[- ]sided|biased?)\b", re.I)

# Measure on the human tagging revisions only. Bot dating edits say
# "Dating maintenance tags: {{POV}}", which would flatter the heuristic badly.
clean = located[~located.first_tag_censored.fillna(False).astype(bool)]
hit = clean.onset_comment.fillna("").str.contains(COMMENT_RE)
print("edit summaries of human tag-adding revisions that mention neutrality: "
      "%.1f%%  (%d/%d)" % (100 * hit.mean(), hit.sum(), len(hit)))
print("so a summary-regex corpus builder would miss %.0f%% of real tag events"
      % (100 * (1 - hit.mean())))
bot_say = located.anchor_comment.fillna("").str.contains("Dating maintenance tags", case=False)
print("(for comparison: %d/%d anchor revisions are bot dating edits)"
      % (bot_say.sum(), len(located)))
print("\n--- examples the heuristic MISSES (silent taggings) ---")
for c in clean.loc[~hit, "onset_comment"].head(8):
    print("  .", (c or "<empty summary>")[:110])
print("\n--- examples it catches ---")
for c in clean.loc[hit, "onset_comment"].head(5):
    print("  .", c[:110])

import os
located.to_csv(os.path.join(OUT_DIR, "full_npov_events_en.csv"), index=False)
events.to_csv(os.path.join(OUT_DIR, "full_npov_localisation_attempts.csv"), index=False)
print("\nFULL RUN DONE:", len(located), "located of", len(events), "attempted")
print("live api calls:", api_stats["calls"], "| bytes:", api_stats["bytes"])
