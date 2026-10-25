#!/usr/bin/env python3
"""
Fine-tuned classifier, stage A — assemble training + lead-time text tables.

The decisive robustness check: instead of an off-the-shelf WNC/LLM scorer, train a
classifier ON OUR OWN tagged-vs-control text (leakage-clean), then re-run the lead-time +
DiD test. If even a trained model shows no drift, the no-drift null is airtight.

Outputs (text = space-joined sentences; dates = matched event onset, for temporal split):
  npov_data/nb_clf_train.jsonl     tag-time text, 962 articles (481 pairs), {pair,title,y,date,text}
  npov_data/nb_clf_leadtime.jsonl  multi-lag text, 421 pairs x {0,30,90,180,365}d, {pair,title,y,lag,revid,text}
"""
import json, pandas as pd

ROOT="."; OUT=ROOT+"/npov_data"
ev=pd.read_csv(OUT+"/full_npov_events_en.csv")
ev["date"]=pd.to_datetime(ev["onset_ts"],errors="coerce",utc=True)
date_of=ev.set_index("title")["date"].to_dict()

def text_of(sents): return " ".join(s.strip() for s in sents if s.strip())

n=0
with open(OUT+"/nb3_sentences.jsonl") as f, open(OUT+"/nb_clf_train.jsonl","w") as g:
    for l in f:
        o=json.loads(l)
        d=date_of.get(o["pair"])
        g.write(json.dumps({"pair":o["pair"],"title":o["title"],"y":o["y"],
                            "date":(str(d) if pd.notna(d) else None),
                            "text":text_of(o["sentences"])})+"\n"); n+=1
print("wrote nb_clf_train.jsonl:", n)

m=0
with open(OUT+"/nb4_sentences.jsonl") as f, open(OUT+"/nb_clf_leadtime.jsonl","w") as g:
    for l in f:
        o=json.loads(l)
        if o.get("revid") is None: continue
        g.write(json.dumps({"pair":o["pair"],"title":o["title"],"y":o["y"],
                            "lag":o["lag"],"revid":o["revid"],
                            "text":text_of(o["sentences"])})+"\n"); m+=1
print("wrote nb_clf_leadtime.jsonl:", m)
