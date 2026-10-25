# Datasheet — NPOV-Onset (English Wikipedia)

A dated benchmark of neutral-point-of-view (NPOV) dispute onsets on English Wikipedia,
with topically and activity-matched never-tagged controls. This datasheet follows the
structure of Gebru et al., *Datasheets for Datasets* (2021).

- **Version:** 1.0
- **Release date:** 2026-10-25
- **Authors:** Mahima Agarwal (`User:AgarwalMahima`), Dev (`Devdoingdev`)
- **Tracking task:** [phab:T428896](https://phabricator.wikimedia.org/T428896)
- **Code:** <https://github.com/mahi01agarwal/npov-drift-detector>
- **License:** CC BY-SA 4.0 (derived from English Wikipedia; see *Distribution*)

---

## Motivation

**For what purpose was the dataset created?**
To study whether Wikipedia articles can be seen *drifting* away from a neutral point of
view *before* a human editor flags them with an NPOV maintenance template — and, more
basically, to provide a ground truth that pins each dispute to the exact revision that
introduced it. Most prior work dates a maintenance template by when a bot stamped it with
a month, or recovers tag additions from edit-summary regexes; both are biased (see
*Collection*). The dataset was built to support a fair, lead-time evaluation: for each
tagged article, when did the tag really appear, and how does the article compare to a
similar article that was never tagged?

**Who created the dataset and on behalf of whom?**
Mahima Agarwal and Dev, as independent research connected to the Wikimedia Phabricator
task [T428896](https://phabricator.wikimedia.org/T428896). It was not commissioned or
funded by the Wikimedia Foundation.

**Who funded the creation of the dataset?**
No external funding. All data was collected read-only through the public MediaWiki Action
API.

---

## Composition

**What do the instances represent?**
Three related tables of English Wikipedia articles and revisions.

1. `full_npov_events_en.csv` — **2,013 NPOV-dispute onset events.** One row per article
   that received an NPOV-family maintenance template, with the template localised to the
   single revision that introduced it.
2. `full_npov_controls_en.csv` — **688 matched control articles.** Each is a never-tagged
   article matched to one event article by shared Wikipedia category and by edit activity,
   giving **688 matched pairs** (a subset of the 2,013 events for which a qualifying
   control was found).
3. `full_npov_localisation_attempts.csv` — **2,166 localisation attempts**, the full log
   from which the 2,013 located events were drawn (localisation success ≈ 93%). Included
   for transparency about the ground-truth construction and its failure modes.

**How many instances are there in total?**
2,013 events, 688 controls (688 matched pairs), 2,166 localisation attempts.

**Does the dataset contain all possible instances or is it a sample?**
It is a sample of English Wikipedia NPOV disputes whose onset fell inside the observation
window **2018-07 to 2026-10** (by `tag_month`; onset timestamps range 2018-07-11 to
2026-10-16) and whose onset revision could be localised. It is not the complete historical
set of NPOV disputes.

**What data does each instance consist of?**
Revision identifiers, timestamps, the editor who introduced the tag, edit summaries, the
template used and its scope, article size, and — for controls — the matching category and
distance. Full field-by-field dictionaries are below. The dataset does **not** ship the
article wikitext itself; every revision is referenced by `revid` and can be re-fetched
from the MediaWiki API (the project's scripts cache these fetches locally).

**Is there a label or target associated with each instance?**
Yes, implicitly: event rows are positive (later tagged) and control rows are negative
(never tagged), joined through `controls.matched_to = events.title`. The intended target
depends on the task — concurrent detection (tagged vs. control at tag time) or lead-time
drift (the same comparison at 30/90/180/365 days before the tag).

**Composition at a glance (events):**

| Field | Breakdown |
|---|---|
| Template | `{{POV}}` 1,330 · `{{NPOV}}` 293 · `{{POV section}}` 245 · `{{Neutrality}}` 91 · `{{Undue weight}}` 15 · `{{POV check}}` 15 · `{{POV statement}}` 9 · `{{Unbalanced}}` 7 |
| Scope | article 1,646 · section 367 |
| Localisation mode | dated 2,011 · any-tag 2 |
| Article size (chars at onset) | median 16,752 · mean 30,905 · range 1,531–518,798 |

**Are there recommended data splits?**
Yes. To avoid leakage, split by **matched pair** (an event and its control stay on the
same side) and/or **temporally** (train on onsets before a cutoff date, test on or after).
The accompanying code uses `GroupKFold` with the pair as the group, and a temporal split,
for exactly this reason.

**Are there errors, sources of noise, or redundancies?**
Yes, and they are the point of several fields:
- ~26% of maintenance templates are **bot-dated**; `anchor_is_bot_dating` flags when the
  dated-anchor revision was a bot run rather than the human who added the tag. The
  localised `onset_revid` corrects for this by binary-searching the real onset.
- ~9% of candidate tags are **left-censored** (`first_tag_censored = True`): a tag already
  present at the start of the search window, so the true onset predates observation.
- A small number of titles are redirects or stubs at older lags; downstream code filters
  snapshots shorter than 30 words or beginning with `#REDIRECT`.

**Is the dataset self-contained, or does it rely on external resources?**
It relies on English Wikipedia: the article text for any revision must be fetched from the
MediaWiki API using the `revid`s provided. Wikipedia revisions are permanent and publicly
addressable, so the references are stable, but the dataset is a set of pointers plus
derived metadata, not a text corpus.

**Does the dataset contain data that might be considered confidential or sensitive?**
No. Everything derives from public Wikipedia edit history. `onset_user` contains editor
usernames (or IP addresses, where an edit was made logged-out) **exactly as they already
appear in public page histories**; no private information is added. See *Collection* for
the privacy stance.

### Data dictionary — `full_npov_events_en.csv`

| Column | Type | Meaning |
|---|---|---|
| `title` | str | Article title (English Wikipedia). |
| `tag_month` | date | Month the maintenance template was dated to (`YYYY-MM-01`). |
| `window_start`, `window_end` | date | Revision search window used to localise the onset. |
| `fetches` | int | Number of API history fetches used to localise this event. |
| `status` | str | Localisation status; `located` for all released rows. |
| `match_mode` | str | `dated` = localised via the template's own date; `any_tag` = fallback to first appearance of any NPOV tag. |
| `window_revs` | int | Revisions examined in the window. |
| `lookback_months` | int | Months searched back from the dated month. |
| `anchor_revid`, `anchor_ts`, `anchor_comment` | int/ts/str | The revision the template's date points to (the "anchor") and its metadata. |
| `anchor_is_bot_dating` | bool | True if the anchor edit is a bot dating the template (not the human who added it). |
| `first_tag_censored` | bool | True if an NPOV tag was already present at `window_start` (true onset predates observation). |
| `onset_revid`, `onset_ts` | int/ts | **The localised onset: the revision that introduced the tag**, found by binary search. |
| `onset_user` | str | Editor who introduced the tag (username or IP, as in public history). |
| `onset_comment` | str | Edit summary of the onset revision. |
| `onset_size` | int | Byte size of the onset revision. |
| `prev_revid` | int | Revision immediately before onset. |
| `predecessor_clean` | bool | True if the predecessor revision carried no NPOV tag (confirms a clean onset boundary). |
| `anchor_lag_revs` | int | Revisions between onset and the dated anchor. |
| `template` | str | NPOV-family template detected (e.g. `pov`, `npov`, `pov section`). |
| `scope` | str | `article` or `section`. |
| `section` | str | Section name, when the tag scopes a section. |
| `template_date` | str | Human date string on the template (e.g. `June 2023`). |
| `article_chars` | int | Character count of the article text at onset. |

### Data dictionary — `full_npov_controls_en.csv`

| Column | Type | Meaning |
|---|---|---|
| `title` | str | Control article title. |
| `pageid` | int | MediaWiki page id. |
| `length` | int | Article length in characters. |
| `n_revs_capped` | int | Revision count (capped during collection). |
| `revs_truncated` | bool | True if the revision count hit the cap. |
| `matched_to` | str | The event `title` this article is a control for (the join key). |
| `match_category` | str | Wikipedia category shared with the event article, used to match topic. |
| `match_distance` | float | Matching distance (lower = closer on the activity/topic criteria). |

### Data dictionary — `full_npov_localisation_attempts.csv`

Same columns as `full_npov_events_en.csv` through `anchor_lag_revs` (it is the upstream
log). Rows where `status = located` and localisation qualified became events; the file is
shipped so the ground-truth construction, including its failures, is auditable.

---

## Collection process

**How was the data acquired?**
Observed, not reported or inferred: every value derives from public English Wikipedia
revision history, retrieved read-only through the
[MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page). The code never logs
in and never edits any wiki.

**What mechanisms or procedures were used?**
Custom Python scripts (released with the code):
1. Candidate NPOV-tagged articles were identified from maintenance-category membership and
   template usage.
2. For each, the dated month was read from the template and used to open a search window;
   the **onset revision was localised by binary search** over the window's history,
   checking for the presence of the tag, rather than trusting the bot's dating or an
   edit-summary regex.
3. Controls were built by sampling never-tagged articles that share a category with the
   event article and have comparable edit activity in the window a detector would examine
   (so controls are not dormant).

**Over what timeframe was the data collected?**
The events span onsets from **July 2018 to October 2026**. Collection and localisation
were performed in 2025–2026.

**Were any ethical review processes conducted?**
No formal IRB review; the data is public, low-risk edit metadata. The collection respects
the API's access terms and is read-only.

**Does the dataset relate to people?**
Only incidentally, through editor usernames in `onset_user` and edit summaries, which are
already public in every article's history. No attempt was made to identify, profile, or
aggregate across editors, and we discourage such use (see *Uses*).

---

## Preprocessing / cleaning / labeling

- **Onset localisation** replaces template dating: the released `onset_revid` is the binary-
  searched revision, not the bot's dated month.
- **Bot-dating and censoring are flagged, not dropped** (`anchor_is_bot_dating`,
  `first_tag_censored`), so users can filter as their analysis requires.
- **Control matching** enforces a shared category and an activity criterion; the released
  controls are the 688 events for which a qualifying match existed.
- Downstream text analysis (in the code, not baked into these tables) filters redirect and
  stub snapshots (`#REDIRECT`, &lt; 30 words).
- The raw, unfiltered localisation log is preserved as a separate file.

**Was the raw data saved in addition to the cleaned data?**
Yes — `full_npov_localisation_attempts.csv` is the upstream log; API responses are cached
locally by the code and regenerate on demand.

---

## Uses

**What tasks could the dataset be used for?**
- Concurrent NPOV-dispute detection (tagged vs. matched control at tag time).
- **Lead-time / drift analysis**: does a text or behaviour signal rise before the tag?
- Studying measurement bias in Wikipedia maintenance templates (bot dating, edit-summary
  recall, dormant controls).
- Methodology work on matched-control construction for observational Wikipedia studies.

**What has the dataset been used for already?**
A seven-stage evaluation (see the code README). Headline finding: NPOV-flagged articles
mostly do **not** drift — they read differently from matched twins a full year before the
tag and stay flat up to it (a static risk factor), with one small, *purely lexical*
pre-onset drift that a bag-of-words model captures as well as a fine-tuned transformer.

**Is there anything about the composition or collection that might affect future uses?**
- The label is "received an NPOV template," a proxy for "a human editor judged it
  non-neutral enough to flag" — it is not a gold neutrality label, and tagging practice
  varies by topic and over time.
- Coverage is English-only and skews toward the recent window; do not treat rates here as
  historical base rates for all of Wikipedia.
- The 688 controls are a matched subset, not a random sample of articles.

**Are there tasks for which the dataset should not be used?**
Do **not** use it to profile, rank, or draw conclusions about individual editors named in
`onset_user` or edit summaries, and do not use it to make automated reverts or editorial
actions on live articles. It is a research benchmark, not a moderation tool.

---

## Distribution

**How will the dataset be distributed?**
Through the GitHub repository, a Zenodo deposit (for a citable DOI, archived from a GitHub
release), and a Hugging Face Dataset (for an in-browser row viewer and `load_dataset`
access).

**Under what license?**
CC BY-SA 4.0. The data is derived from English Wikipedia content, which is CC BY-SA
licensed; this dataset inherits that license and attributes Wikipedia as the source. The
accompanying **code** is MIT (see the repository `LICENSE`).

**Are there IP or other restrictions?**
Attribution to English Wikipedia and share-alike are required by CC BY-SA 4.0. No
additional restrictions.

---

## Maintenance

**Who will support/host/maintain the dataset?**
The authors, via the GitHub repository. Issues and corrections are handled there.

**How can the dataset be updated?**
The collection scripts are released; the dataset can be regenerated or extended (e.g. more
controls, other language editions) by re-running them. Versioned releases will be tagged on
GitHub and mirrored to Zenodo; the DOI resolves to a specific version.

**Will older versions continue to be supported?**
Zenodo preserves each released version under its own DOI.

**If others want to extend/contribute, is there a mechanism?**
Pull requests and issues on GitHub. Because every instance is a public Wikipedia revision
id, contributions are verifiable against the source.

---

*This datasheet describes data derived from English Wikipedia and is itself released under
CC BY-SA 4.0.*
