#!/usr/bin/env python3
"""Expanded classifier training table from nb3_sentences_expanded.jsonl (all matched pairs
after the control expansion). Lead-time panel is unchanged (nb_clf_leadtime.jsonl, 421 pairs).
Outputs npov_data/nb_clf_train_expanded.jsonl {pair,title,y,date,text}."""
import json, pandas as pd
OUT="npov_data"
ev=pd.read_csv(OUT+"/full_npov_events_en.csv"); ev["date"]=pd.to_datetime(ev["onset_ts"],errors="coerce",utc=True)
date_of=ev.set_index("title")["date"].to_dict()
def text_of(s): return " ".join(x.strip() for x in s if x.strip())
n=0
with open(OUT+"/nb3_sentences_expanded.jsonl") as f, open(OUT+"/nb_clf_train_expanded.jsonl","w") as g:
    for l in f:
        o=json.loads(l); d=date_of.get(o["pair"])
        g.write(json.dumps({"pair":o["pair"],"title":o["title"],"y":o["y"],
                            "date":(str(d) if pd.notna(d) else None),"text":text_of(o["sentences"])})+"\n"); n+=1
print("wrote nb_clf_train_expanded.jsonl:",n)
