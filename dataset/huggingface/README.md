---
license: cc-by-sa-4.0
language:
- en
pretty_name: NPOV-Onset
size_categories:
- 1K<n<10K
source_datasets:
- original
annotations_creators:
- found
task_categories:
- text-classification
tags:
- wikipedia
- npov
- neutral-point-of-view
- content-moderation
- maintenance-templates
- lead-time
- benchmark
configs:
- config_name: events
  data_files: full_npov_events_en.csv
- config_name: controls
  data_files: full_npov_controls_en.csv
- config_name: localisation_attempts
  data_files: full_npov_localisation_attempts.csv
---

# NPOV-Onset

A dated benchmark of **neutral-point-of-view (NPOV) dispute onsets** on English Wikipedia,
with topically and activity-matched never-tagged controls.

For each of ~2,000 articles that received an NPOV maintenance template, the onset is
localised **by binary search to the exact revision that introduced the tag** — correcting
for the ~26% of templates that are only bot-dated and for edit-summary regexes that recover
barely half of human tag additions. Each event is paired with a never-tagged article that
shares a Wikipedia category and has comparable edit activity, so the comparison is fair.

- **Code & full documentation:** <https://github.com/mahi01agarwal/npov-drift-detector>
- **Datasheet** (Gebru et al. format): see `DATASHEET.md` in the code repository
- **Tracking task:** [phab:T428896](https://phabricator.wikimedia.org/T428896)
- **License:** CC BY-SA 4.0 (derived from English Wikipedia)

## Configurations

| Config | Rows | What it is |
|---|---|---|
| `events` | 2,013 | NPOV-dispute onsets, each pinned to its onset revision |
| `controls` | 688 | Matched never-tagged articles (688 matched pairs; join on `matched_to`) |
| `localisation_attempts` | 2,166 | The full upstream localisation log (≈93% located), for auditing the ground truth |

## Usage

```python
from datasets import load_dataset

events   = load_dataset("<your-namespace>/npov-onset", "events",   split="train")
controls = load_dataset("<your-namespace>/npov-onset", "controls", split="train")

# Reconstruct matched pairs (one control per event, join on title):
ctrl_by_event = {c["matched_to"]: c for c in controls}
pairs = [(e, ctrl_by_event[e["title"]]) for e in events if e["title"] in ctrl_by_event]
print(len(pairs), "matched pairs")   # 688
```

The dataset ships **metadata and revision ids, not article text**. To get the prose for any
snapshot, fetch the `onset_revid` (or an earlier revision for lead-time work) from the
[MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page); the repository's
scripts do this and cache the results.

## Key fields

**events** — `title`, `onset_revid`/`onset_ts` (the localised onset), `onset_user`,
`onset_comment`, `template` (`pov`, `npov`, `pov section`, …), `scope` (article/section),
`anchor_is_bot_dating` (flag: anchor was a bot dating the template), `first_tag_censored`
(flag: tag predates the observation window), `article_chars`.
**controls** — `title`, `pageid`, `length`, `matched_to` (the event it controls for),
`match_category`, `match_distance`.

See the repository `DATASHEET.md` for the complete data dictionary, collection process,
recommended splits (by matched pair and/or temporally, to avoid leakage), and intended-use
limits.

## Intended use and limits

Built for **concurrent detection** and **lead-time / drift** analysis of NPOV disputes, and
for studying measurement bias in Wikipedia maintenance templates. The label is "received an
NPOV template" — a proxy for an editor's judgment, not a gold neutrality label. English
only; the 688 controls are a matched subset, not a random sample.

**Do not** use this to profile or draw conclusions about individual editors named in
`onset_user`/edit summaries (these are already public in page history), or to drive
automated edits on live articles.

## Citation

```bibtex
@misc{npov_onset_2026,
  title        = {NPOV-Onset: a dated benchmark of neutral-point-of-view dispute onsets on English Wikipedia},
  author       = {Agarwal, Mahima and Dev},
  year         = {2026},
  howpublished = {\url{https://github.com/mahi01agarwal/npov-drift-detector}},
  note         = {CC BY-SA 4.0; Wikimedia Phabricator T428896}
}
```
