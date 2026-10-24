#!/usr/bin/env python3
"""
Direction axis, stage A  —  build bounded judge inputs from section snapshots.

New construct: ONE-SIDEDNESS / DUE WEIGHT (are relevant viewpoints represented
proportionally?), deliberately ORTHOGONAL to the NB3/NB4 subjectivity-of-tone axis.
This is the GWeld "slant" tangent applied to our onset-dated, labeled, temporal benchmark.

Turns each cached snapshot (npov_data/nb4_sections.jsonl) into a compact, token-bounded
text an LLM judge can read: title + full lead + ALL section headings (structure reveals
what is covered vs omitted, e.g. a "Criticism" section) + an even body sample across the
non-lead sections. No model call here; deterministic and free.

Output: npov_data/nb4_direction_inputs.jsonl  (one row per snapshot, keyed pair/y/lag/revid)
"""
import json, os

ROOT = "."; OUT = ROOT + "/npov_data"
SRC  = OUT + "/nb4_sections.jsonl"
DST  = OUT + "/nb4_direction_inputs.jsonl"

MAX_BODY_SENTS = 60      # cap body sample (lead is always kept in full)
MAX_SENT_CHARS = 400     # drop pathological long "sentences" (tables, refs)

def even_sample(items, k):
    n = len(items)
    if n <= k: return items
    step = n / k
    return [items[int(i * step)] for i in range(k)]

def build(o):
    secs = o.get("sections", [])
    lead, headings, body = "", [], []
    for s in secs:
        h = s.get("heading", "")
        sents = [x.strip() for x in s.get("sentences", []) if 0 < len(x.strip()) <= MAX_SENT_CHARS]
        if h == "(lead)":
            lead = " ".join(sents)
        else:
            headings.append(h)
            body.extend([(h, x) for x in sents])
    body = even_sample(body, MAX_BODY_SENTS)
    # group sampled body back under its heading, in order
    body_by_h, order = {}, []
    for h, x in body:
        if h not in body_by_h: body_by_h[h] = []; order.append(h)
        body_by_h[h].append(x)
    parts = ["TITLE: " + o["title"], "", "LEAD:", lead or "(none)", "",
             "SECTION HEADINGS (in order): " + (" | ".join(headings) if headings else "(none)"), ""]
    if order:
        parts.append("BODY EXCERPTS (sampled, grouped by section):")
        for h in order:
            parts.append("== %s ==" % h)
            parts.extend(body_by_h[h])
            parts.append("")
    text = "\n".join(parts).strip()
    return text, len(headings), len(body)

def main():
    n = 0
    with open(SRC) as f, open(DST, "w") as g:
        for line in f:
            o = json.loads(line)
            if o.get("revid") is None:   # missing snapshot (article didn't exist that far back)
                continue
            text, nh, nb = build(o)
            g.write(json.dumps({"title": o["title"], "pair": o["pair"], "y": o["y"],
                                "lag": o["lag"], "revid": o["revid"],
                                "n_headings": nh, "n_body_sents": nb,
                                "judge_input": text}) + "\n")
            n += 1
    print("wrote %d judge inputs -> %s" % (n, DST))

if __name__ == "__main__":
    main()
