#!/usr/bin/env python3
"""
Minimal NPOV-Onset loader: read the three tables, reconstruct the 688 matched pairs,
and show how to pull the article text for a snapshot from the MediaWiki API.

Works directly on the CSVs in this folder (no `datasets` install needed). The Hugging Face
equivalent is one `load_dataset(...)` call per config — see the dataset card.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    events = load("full_npov_events_en.csv")
    controls = load("full_npov_controls_en.csv")

    # one control per event, joined on the event title
    ctrl_by_event = {c["matched_to"]: c for c in controls}
    pairs = [(e, ctrl_by_event[e["title"]]) for e in events if e["title"] in ctrl_by_event]
    print(f"{len(events)} events, {len(controls)} controls, {len(pairs)} matched pairs")

    # measurement-bias flags you will want to filter on
    bot = sum(1 for e in events if e["anchor_is_bot_dating"] == "True")
    cens = sum(1 for e in events if e["first_tag_censored"] == "True")
    print(f"bot-dated anchors: {bot} ({bot/len(events):.0%}); "
          f"left-censored: {cens} ({cens/len(events):.0%})")

    e, c = pairs[0]
    onset_revid = int(float(e["onset_revid"]))  # stored as float in CSV; revids are ints
    print(f"\nexample pair:\n  event   {e['title']!r} tagged {e['template']} "
          f"at rev {onset_revid} ({e['onset_ts']})\n"
          f"  control {c['title']!r} (shares {c['match_category']})")

    # To fetch the prose for any snapshot, resolve the revid against the API, e.g.:
    #   https://en.wikipedia.org/w/api.php?action=parse&oldid=<onset_revid>
    #       &prop=wikitext&format=json
    # For lead-time work, pick the revision closest to N days before onset_ts.
    print(f"\ntext for the onset snapshot: "
          f"https://en.wikipedia.org/w/api.php?action=parse&oldid={onset_revid}"
          f"&prop=wikitext&format=json")


if __name__ == "__main__":
    main()
