#!/usr/bin/env python3
"""
Fine-tuned classifier, stage B — the decisive robustness check.

Train an ON-TASK text classifier (fine-tuned RoBERTa) to tell onset tag-time text from
matched control text, LEAKAGE-CLEAN, then re-run the lead-time + DiD test on its scores.
If even a trained model shows no drift, the no-drift null is airtight (not an artifact of
the off-the-shelf WNC subjectivity model or the LLM one-sidedness judge).

Two modes:
  --mode cv        StratifiedGroupKFold(group=pair): every article scored out-of-fold
                   (no pair in both train and its own eval). Uses all 481 pairs.
  --mode temporal  train on pairs with onset < --cutoff, test on later pairs (stronger,
                   guards against period/topic drift as well as article leakage).

Concurrent read: unpaired AUC + matched win-rate at lag 0 (does on-task text beat the
0.76 metadata floor / the +0.034 WNC signal?).
Drift read: per (pair,lag) P(tagged); matched win-rate per lag + DiD 365d->0d.

Inputs : npov_data/nb_clf_train.jsonl, npov_data/nb_clf_leadtime.jsonl
Outputs: npov_data/nb_clf_oof.csv, nb_clf_leadtime_preds.csv, nb_clf_curve_<mode>.csv
"""
import os, json, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold

ROOT="."; OUT=ROOT+"/npov_data"
SEED=20261025
torch.manual_seed(SEED); np.random.seed(SEED)
DEV="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load(train_file="nb_clf_train.jsonl"):
    tr=pd.DataFrame(json.loads(l) for l in open(OUT+"/"+train_file))
    lt=pd.DataFrame(json.loads(l) for l in open(OUT+"/nb_clf_leadtime.jsonl"))
    tr["date"]=pd.to_datetime(tr["date"],errors="coerce",utc=True)
    return tr, lt

class DS(torch.utils.data.Dataset):
    def __init__(self,texts,labels,tok,maxlen):
        self.e=tok(list(texts),truncation=True,max_length=maxlen,padding=False); self.y=list(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return {"input_ids":self.e["input_ids"][i],
                                     "attention_mask":self.e["attention_mask"][i],"labels":self.y[i]}

def collate(tok):
    def f(b):
        ids=[x["input_ids"] for x in b]; am=[x["attention_mask"] for x in b]; y=[x["labels"] for x in b]
        m=max(len(i) for i in ids)
        pad=tok.pad_token_id
        ids=[i+[pad]*(m-len(i)) for i in ids]; am=[a+[0]*(m-len(a)) for a in am]
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(am),"labels":torch.tensor(y)}
    return f

def train_one(texts,labels,args,tok):
    mdl=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2).to(DEV)
    dl=DataLoader(DS(texts,labels,tok,args.maxlen),batch_size=args.bs,shuffle=True,collate_fn=collate(tok))
    opt=torch.optim.AdamW(mdl.parameters(),lr=args.lr)
    steps=len(dl)*args.epochs
    sch=get_linear_schedule_with_warmup(opt,int(0.06*steps),steps)
    scaler=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda"))
    mdl.train()
    for ep in range(args.epochs):
        for b in dl:
            b={k:v.to(DEV) for k,v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda" if DEV=="cuda" else "cpu",enabled=(DEV=="cuda")):
                out=mdl(**b); loss=out.loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    return mdl

@torch.no_grad()
def predict(mdl,texts,args,tok):
    mdl.eval(); probs=[]
    dl=DataLoader(DS(texts,[0]*len(texts),tok,args.maxlen),batch_size=args.bs*2,shuffle=False,collate_fn=collate(tok))
    for b in dl:
        b={k:v.to(DEV) for k,v in b.items() if k!="labels"}
        with torch.autocast(device_type="cuda" if DEV=="cuda" else "cpu",enabled=(DEV=="cuda")):
            logit=mdl(**b).logits
        probs.append(torch.softmax(logit.float(),-1)[:,1].cpu().numpy())
    return np.concatenate(probs)

def unpaired_auc(pos,neg):
    allv=np.concatenate([pos,neg]); r=pd.Series(allv).rank().values
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def analyse(concurrent, ltpred, tag):
    rng=np.random.default_rng(SEED)
    # concurrent: unpaired AUC + matched winrate at lag 0
    pos=concurrent[concurrent.y==1].p.values; neg=concurrent[concurrent.y==0].p.values
    auc=unpaired_auc(pos,neg)
    piv=concurrent.pivot_table(index="pair",columns="y",values="p",aggfunc="first").dropna()
    wr0=((piv[1]-piv[0])>0).mean()+0.5*((piv[1]-piv[0])==0).mean()
    print(f"\n[{tag}] CONCURRENT  unpaired AUC={auc:.3f}  matched win-rate(lag0)={wr0:.3f}  (n_pairs={len(piv)})")
    # drift: matched winrate per lag + DiD
    lags=sorted(ltpred.lag.unique()); rows=[]
    W={}
    for lag in lags:
        s=ltpred[ltpred.lag==lag]
        p=s.pivot_table(index="pair",columns="y",values="p",aggfunc="first").dropna()
        if len(p)<5: continue
        diff=(p[1]-p[0]).values
        wr=(diff>0).mean()+0.5*(diff==0).mean()
        bw=np.array([ (lambda d:(d>0).mean()+0.5*(d==0).mean())(diff[rng.integers(0,len(diff),len(diff))]) for _ in range(5000)])
        rows.append({"lag":lag,"n_pairs":len(p),"winrate":wr,"wr_lo":np.percentile(bw,2.5),"wr_hi":np.percentile(bw,97.5),"mean_gap":diff.mean()})
        W[lag]=p
    print(f"[{tag}] DRIFT  matched win-rate (tagged P>control P) per lag:")
    for r in rows:
        star=" *" if r["wr_lo"]>0.5 else ""
        print("   %4dd  n=%d  winrate=%.3f [%.3f,%.3f]  gap=%+.4f%s"%(r["lag"],r["n_pairs"],r["winrate"],r["wr_lo"],r["wr_hi"],r["mean_gap"],star))
    far,near=max(W),min(W)
    common=W[far].index.intersection(W[near].index)
    if len(common)>=5:
        dP=(W[near].loc[common,1]-W[far].loc[common,1]).values
        dC=(W[near].loc[common,0]-W[far].loc[common,0]).values
        did=dP-dC; n=len(did)
        bd=np.array([did[rng.integers(0,n,n)].mean() for _ in range(5000)])
        lo,hi=np.percentile(bd,2.5),np.percentile(bd,97.5)
        verd="DRIFT: tagged rose MORE" if lo>0 else ("tagged rose LESS" if hi<0 else "NO DRIFT (CI spans 0)")
        print(f"[{tag}] DRIFT DiD {far}d->{near}d (tagged change minus twin, n={n}): {did.mean():+.4f} [{lo:.4f},{hi:.4f}] => {verd}")
    pd.DataFrame(rows).to_csv(OUT+f"/nb_clf_curve_{tag}.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="roberta-base")
    ap.add_argument("--mode",default="cv",choices=["cv","temporal","expanded"])
    ap.add_argument("--train-file",default="nb_clf_train.jsonl")
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--bs",type=int,default=16)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--maxlen",type=int,default=512)
    ap.add_argument("--cutoff",default="2024-06-01")
    a=ap.parse_args()
    print("device",DEV,"| model",a.model,"| mode",a.mode,flush=True)
    tok=AutoTokenizer.from_pretrained(a.model)
    tr,lt=load(a.train_file)
    conc=[]; ltp=[]
    if a.mode=="expanded":
        panel=set(lt.pair)                         # 421 drift panel = fully held out
        trn=tr[~tr.pair.isin(panel)]               # train on big non-panel set
        print("-- expanded: train %d (non-panel) | drift eval on %d panel pairs"%(len(trn),len(panel)),flush=True)
        mdl=train_one(trn.text,trn.y,a,tok)
        sub=lt[lt.pair.isin(panel)].copy(); sub["p"]=predict(mdl,sub.text.tolist(),a,tok)
        ltp.append(sub[["pair","title","y","lag","p"]])
        conc.append(sub[sub.lag==0][["pair","title","y","p"]])   # concurrent read = panel lag0
    if a.mode=="cv":
        gkf=GroupKFold(n_splits=a.folds)
        for fi,(tri,tei) in enumerate(gkf.split(tr,tr.y,groups=tr.pair)):
            trn=tr.iloc[tri]; tst=tr.iloc[tei]; test_pairs=set(tst.pair)
            print(f"-- fold {fi+1}/{a.folds}: train {len(trn)} test {len(tst)} ({len(test_pairs)} pairs)",flush=True)
            mdl=train_one(trn.text,trn.y,a,tok)
            tst=tst.copy(); tst["p"]=predict(mdl,tst.text.tolist(),a,tok); conc.append(tst[["pair","title","y","p"]])
            sub=lt[lt.pair.isin(test_pairs)].copy()
            if len(sub): sub["p"]=predict(mdl,sub.text.tolist(),a,tok); ltp.append(sub[["pair","title","y","lag","p"]])
            del mdl; torch.cuda.empty_cache()
    elif a.mode=="temporal":
        cut=pd.Timestamp(a.cutoff,tz="UTC")
        trn=tr[tr.date<cut]; tst=tr[tr.date>=cut]; test_pairs=set(tst.pair)
        print(f"-- temporal: train {len(trn)} (<{a.cutoff}) test {len(tst)} (>={a.cutoff}, {len(test_pairs)} pairs)",flush=True)
        mdl=train_one(trn.text,trn.y,a,tok)
        tst=tst.copy(); tst["p"]=predict(mdl,tst.text.tolist(),a,tok); conc.append(tst[["pair","title","y","p"]])
        sub=lt[lt.pair.isin(test_pairs)].copy()
        if len(sub): sub["p"]=predict(mdl,sub.text.tolist(),a,tok); ltp.append(sub[["pair","title","y","lag","p"]])
    conc=pd.concat(conc); ltp=pd.concat(ltp)
    conc.to_csv(OUT+"/nb_clf_oof.csv",index=False); ltp.to_csv(OUT+"/nb_clf_leadtime_preds.csv",index=False)
    analyse(conc,ltp,a.mode)
    print("\nDONE_"+a.mode,flush=True)

if __name__=="__main__": main()
