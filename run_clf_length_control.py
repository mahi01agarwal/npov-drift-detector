#!/usr/bin/env python3
"""
NB-clf length-control — is the classifier's pre-onset DRIFT genuine content drift, or a proxy
for article GROWTH (tagged articles grow/churn more toward onset = NB2 attention via the text)?
Regress per-pair drift_p (DiD of classifier score) on growth_diff (DiD of length); the INTERCEPT
is the length-controlled drift. Intercept CI>0 => drift not explained by growth.
"""
import json, numpy as np, pandas as pd
OUT="npov_data"; rng=np.random.default_rng(20261025); FAR,NEAR=365,0
pred=pd.read_csv(OUT+"/nb_clf_leadtime_preds.csv")
L={}
for l in open(OUT+"/nb_clf_leadtime.jsonl"):
    o=json.loads(l); L[(o["pair"],o["y"],o["lag"])]=len(o["text"].split())
pred["L"]=[L.get((r.pair,r.y,r.lag),np.nan) for r in pred.itertuples()]

def delta(yv,col):
    t=pred[pred.y==yv].pivot_table(index="pair",columns="lag",values=col,aggfunc="first")
    return (t[NEAR]-t[FAR]) if (FAR in t and NEAR in t) else None

df=pd.DataFrame({"pt":delta(1,"p"),"pc":delta(0,"p"),"lt":delta(1,"L"),"lc":delta(0,"L")}).dropna()
drift_p=(df.pt-df.pc).values; grow_d=(df.lt-df.lc).values; n=len(df)
print("pairs with 365&0 both sides:",n)
print("\n--- descriptives ---")
print("mean drift_p (DiD classifier score)        = %+.4f"%drift_p.mean())
print("tagged  mean length change 365->0 = %+.1f words"%df.lt.mean())
print("control mean length change 365->0 = %+.1f words"%df.lc.mean())
print("mean growth_diff (tagged-control)          = %+.1f words"%grow_d.mean())
print("corr(drift_p, growth_diff)                 = %.3f"%np.corrcoef(drift_p,grow_d)[0,1])

def ols(y,x): x1=np.c_[np.ones(len(x)),x]; b,_,_,_=np.linalg.lstsq(x1,y,rcond=None); return b
xs=(grow_d-grow_d.mean())/(grow_d.std()+1e-9)
a,b=ols(drift_p,xs)
bb=np.array([ (lambda i: ols(drift_p[i], (grow_d[i]-grow_d[i].mean())/(grow_d[i].std()+1e-9)))(rng.integers(0,n,n)) for _ in range(5000)])
alo,ahi=np.percentile(bb[:,0],[2.5,97.5]); blo,bhi=np.percentile(bb[:,1],[2.5,97.5])
r2=1-((drift_p-(a+b*xs))**2).sum()/((drift_p-drift_p.mean())**2).sum()
print("\n--- length-controlled regression (drift_p ~ standardized growth_diff) ---")
print("INTERCEPT = length-controlled drift = %+.4f  [%.4f, %.4f]"%(a,alo,ahi))
print("slope per +1 SD growth              = %+.4f  [%.4f, %.4f]   R2=%.3f"%(b,blo,bhi,r2))
print("=> drift", "SURVIVES length control (genuine beyond growth)" if alo>0 else
      ("EXPLAINED by growth (CI spans 0)" if ahi>0 else "negative"))

thr=np.quantile(np.abs(grow_d),0.33); m=np.abs(grow_d)<=thr; sub=drift_p[m]
bm=np.array([sub[rng.integers(0,len(sub),len(sub))].mean() for _ in range(5000)])
print("\n--- restricted to LOW differential-growth pairs (|growth_diff|<=%.0f w, n=%d) ---"%(thr,m.sum()))
print("mean drift_p = %+.4f  [%.4f, %.4f]  => %s"%(sub.mean(),np.percentile(bm,2.5),np.percentile(bm,97.5),
      "drift persists without growth" if np.percentile(bm,2.5)>0 else "null when growth removed"))
