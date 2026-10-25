#!/usr/bin/env python3
"""Overnight: analyse the EXPANDED-data classifier run and write OVERNIGHT_SUMMARY.md.
Robust to missing inputs (reports what exists). Compares expanded vs the 481-pair baseline."""
import json, os, numpy as np, pandas as pd
from datetime import datetime
OUT="npov_data"; rng=np.random.default_rng(20261025); FAR,NEAR=365,0
lines=[]; p=lines.append
def ex(f): return os.path.exists(f)

def unpaired_auc(pos,neg):
    allv=np.concatenate([pos,neg]); r=pd.Series(allv).rank().values
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def did(dfp,label):
    def delta(yv):
        t=dfp[dfp.y==yv].pivot_table(index="pair",columns="lag",values="p",aggfunc="first")
        return (t[NEAR]-t[FAR]) if (FAR in t and NEAR in t) else None
    dp,dc=delta(1),delta(0)
    if dp is None or dc is None: return None
    both=dp.dropna().index.intersection(dc.dropna().index); d=(dp.loc[both]-dc.loc[both]).values; n=len(d)
    bd=np.array([d[rng.integers(0,n,n)].mean() for _ in range(5000)]); lo,hi=np.percentile(bd,[2.5,97.5])
    c0=dfp[dfp.lag==0]; auc=unpaired_auc(c0[c0.y==1].p.dropna().values,c0[c0.y==0].p.dropna().values)
    return dict(label=label,n=n,did=d.mean(),lo=lo,hi=hi,auc=auc)

p("# Overnight run — expanded-data classifier\n")
p("_Generated %s_\n"%datetime.now().strftime("%Y-%m-%d %H:%M"))

# dataset sizes
nc = (sum(1 for _ in open(OUT+"/full_npov_controls_en.csv"))-1) if ex(OUT+"/full_npov_controls_en.csv") else "?"
ne = sum(1 for _ in open(OUT+"/nb3_sentences_expanded.jsonl")) if ex(OUT+"/nb3_sentences_expanded.jsonl") else 0
ntr= sum(1 for _ in open(OUT+"/nb_clf_train_expanded.jsonl")) if ex(OUT+"/nb_clf_train_expanded.jsonl") else 0
p("## Dataset")
p("- Controls after expansion: **%s** (was 481)"%nc)
p("- Expanded tag-time articles fetched: **%d**"%ne)
p("- Expanded classifier training rows: **%d** (was 962)\n"%ntr)

# load leadtime text for lexical + stub flag
have_lt_text = ex(OUT+"/nb_clf_leadtime.jsonl")
if have_lt_text:
    lt=pd.DataFrame(json.loads(l) for l in open(OUT+"/nb_clf_leadtime.jsonl"))
    lt["nwords"]=lt.text.str.split().str.len()
    lt["is_stub"]=(lt.nwords<30)|lt.text.str.contains("#REDIRECT",case=False)

# RoBERTa expanded drift
p("## RoBERTa — expanded model (trained on big non-panel set, drift eval on 421 panel)")
rob_f=OUT+"/nb_clf_leadtime_preds_expanded.csv"
if ex(rob_f):
    rob=pd.read_csv(rob_f)
    if have_lt_text:
        rob=rob.merge(lt[["pair","y","lag","is_stub"]],on=["pair","y","lag"],how="left")
        r=did(rob[~rob.is_stub.fillna(False)],"RoBERTa expanded (no stubs)")
    else:
        r=did(rob,"RoBERTa expanded")
    if r: p("- DiD 365→0 = **%+.4f [%.4f, %.4f]** | concurrent AUC %.3f | n=%d  → %s"%(
            r["did"],r["lo"],r["hi"],r["auc"],r["n"],"DRIFT" if r["lo"]>0 else "no drift (CI spans 0)"))
    cf=OUT+"/nb_clf_curve_expanded.csv"
    if ex(cf):
        c=pd.read_csv(cf); 
        if "lag" in c and "winrate" in c:
            p("- per-lag win-rate: "+" ".join("%dd=%.3f"%(x.lag,x.winrate) for x in c.sort_values("lag",ascending=False).itertuples()))
else:
    p("- _expanded RoBERTa predictions not found — dev-4 retrain may have failed; check clf_expanded.log_")

# Lexical expanded
p("\n## Lexical (TF-IDF+LogReg) — expanded, same held-out panel")
if ex(OUT+"/nb_clf_train_expanded.jsonl") and have_lt_text:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        tr=pd.DataFrame(json.loads(l) for l in open(OUT+"/nb_clf_train_expanded.jsonl"))
        panel=set(lt.pair); trn=tr[~tr.pair.isin(panel)]
        vec=TfidfVectorizer(sublinear_tf=True,min_df=3,max_df=0.9,ngram_range=(1,2),max_features=50000)
        clf=LogisticRegression(max_iter=2000,C=4.0).fit(vec.fit_transform(trn.text),trn.y)
        sub=lt[lt.pair.isin(panel)].copy(); sub["p"]=clf.predict_proba(vec.transform(sub.text))[:,1]
        r=did(sub[~sub.is_stub],"lexical expanded (no stubs)")
        if r: p("- DiD 365→0 = **%+.4f [%.4f, %.4f]** | concurrent AUC %.3f | train %d  → %s"%(
                r["did"],r["lo"],r["hi"],r["auc"],len(trn),"DRIFT" if r["lo"]>0 else "no drift"))
    except Exception as e:
        p("- lexical analysis failed: %s"%e)
else:
    p("- _inputs missing_")

p("\n## Baseline (481 pairs, for comparison)")
p("- RoBERTa temporal DiD +0.0081 [0.0023,0.0143] (AUC 0.653); lexical +0.0134 [0.0059,0.0217] (AUC 0.740)")
p("- Verdict so far: small real LEXICAL drift; bag-of-words ≈ RoBERTa at 962 examples.")
p("\n## What to check on the bigger data")
p("- Did RoBERTa's concurrent AUC overtake lexical with more training data?")
p("- Did RoBERTa's DiD grow relative to lexical (= non-lexical signal emerging)?")
p("- If RoBERTa now clearly beats lexical on drift → there IS semantic drift beyond vocabulary.")

open("OVERNIGHT_SUMMARY.md","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
print("\nwrote OVERNIGHT_SUMMARY.md")
