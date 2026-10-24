#!/usr/bin/env python3
"""
Direction axis, stage C  —  one-sidedness lead-time curve (the GWeld tangent).

Does an article's ONE-SIDEDNESS (due-weight imbalance) DRIFT upward before the human NPOV
tag, or is it -- like subjectivity-of-tone (NB4) and like GWeld's slant -- a STATIC trait?

Identical two readings as NB4/NB4b so results drop into the paper unchanged:
  [A] MATCHED win-rate per lag  -- is the tagged article more one-sided than its twin, and
                                   is that gap already there a year out (static risk factor)?
  [B] DiD 365d -> 0d            -- does the tagged side's one-sidedness rise MORE than its
                                   matched twin's over the identical window (drift, net of aging)?

Primary feature one_sidedness; missing_other_side / framing_partiality are robustness axes.

Input:  npov_data/nb4_direction.csv     (from score_direction.py)
Output: npov_data/nb4_direction_curve.csv
"""
import numpy as np, pandas as pd

ROOT="."; OUT=ROOT+"/npov_data"
SEED=20261024; rng=np.random.default_rng(SEED)
FEATS=["one_sidedness","missing_other_side","framing_partiality"]

d=pd.read_csv(OUT+"/nb4_direction.csv")
d=d[d.revid.notna()].copy()
lags=sorted(d.lag.unique())
print("rows %d | lags %s | pairs %d" % (len(d),lags,d.pair.nunique()))

def wide(df,feat):
    pos=df[df.y==1][["pair","lag",feat]].rename(columns={feat:"pos"})
    ctl=df[df.y==0][["pair","lag",feat]].rename(columns={feat:"ctl"})
    return pos.merge(ctl,on=["pair","lag"],how="inner").dropna()

def winrate_gap(sub):
    diff=sub.pos.values-sub.ctl.values
    return (diff>0).mean()+0.5*(diff==0).mean(), diff.mean()

def boot_pair(sub,fn,n=5000):
    pairs=sub.pair.values
    idx={p:np.where(pairs==p)[0] for p in np.unique(pairs)}
    up=np.unique(pairs)
    return np.array([fn(sub.iloc[np.concatenate([idx[p] for p in rng.choice(up,len(up),True)])])
                     for _ in range(n)])

rows=[]
for feat in FEATS:
    W=wide(d,feat)
    if W.empty: continue
    cnt=W.groupby("pair").lag.nunique()
    balanced=set(cnt[cnt==len(lags)].index)
    for panel,keep in [("balanced",balanced),("surviving",None)]:
        for lag in lags:
            sub=W[W.lag==lag]
            if keep is not None: sub=sub[sub.pair.isin(keep)]
            if len(sub)<5: continue
            wr,gp=winrate_gap(sub)
            bw=boot_pair(sub,lambda s:winrate_gap(s)[0])
            bg=boot_pair(sub,lambda s:winrate_gap(s)[1])
            rows.append({"feat":feat,"test":"matched_winrate","panel":panel,"lag":lag,
                         "n_pairs":len(sub),"winrate":wr,
                         "wr_lo":np.percentile(bw,2.5),"wr_hi":np.percentile(bw,97.5),
                         "mean_gap":gp,"gap_lo":np.percentile(bg,2.5),"gap_hi":np.percentile(bg,97.5)})
    far,near=max(lags),min(lags)
    def side_delta(yv):
        t=d[(d.y==yv)&d.pair.isin(balanced)][["pair","lag",feat]]
        piv=t.pivot_table(index="pair",columns="lag",values=feat,aggfunc="first")
        if far not in piv.columns or near not in piv.columns: return None
        return (piv[near]-piv[far]).rename("d")
    dp,dc=side_delta(1),side_delta(0)
    if dp is not None and dc is not None:
        bp=dp.dropna().index.intersection(dc.dropna().index)
        did=(dp.loc[bp]-dc.loc[bp]).values; n=len(did)
        if n>=5:
            win=(did>0).mean()+0.5*(did==0).mean()
            bd=np.array([did[rng.integers(0,n,n)].mean() for _ in range(5000)])
            bw=np.array([(lambda x:(x>0).mean()+0.5*(x==0).mean())(did[rng.integers(0,n,n)]) for _ in range(5000)])
            rows.append({"feat":feat,"test":"within_drift_DiD_365to0","panel":"balanced","lag":-1,
                         "n_pairs":n,"winrate":win,"wr_lo":np.percentile(bw,2.5),"wr_hi":np.percentile(bw,97.5),
                         "mean_gap":float(did.mean()),"gap_lo":np.percentile(bd,2.5),"gap_hi":np.percentile(bd,97.5)})

res=pd.DataFrame(rows); res.to_csv(OUT+"/nb4_direction_curve.csv",index=False)
for feat in FEATS:
    print("\n==================  %s  ==================" % feat)
    sub=res[(res.feat==feat)&(res.test=="matched_winrate")&(res.panel=="balanced")]
    if len(sub):
        print("[A] MATCHED win-rate (tagged more one-sided than twin), balanced panel")
        print("  %5s %7s %9s %22s %12s" % ("lag","pairs","winrate","95% CI","mean_gap"))
        for r in sub.sort_values("lag").itertuples(index=False):
            star=" *" if r.wr_lo>0.5 else ""
            print("  %4dd %7d %9.3f   [%.3f, %.3f]   %+.4f%s"%(r.lag,r.n_pairs,r.winrate,r.wr_lo,r.wr_hi,r.mean_gap,star))
    w=res[(res.feat==feat)&(res.test=="within_drift_DiD_365to0")]
    if len(w):
        r=w.iloc[0]
        verd="DRIFT: tagged rose MORE" if r.gap_lo>0 else ("tagged rose LESS" if r.gap_hi<0 else "NO DRIFT (CI spans 0)")
        print("[B] DiD 365d -> 0d, tagged change minus twin's (n=%d):  %+.4f  [%.4f, %.4f]  => %s"
              %(r.n_pairs,r.mean_gap,r.gap_lo,r.gap_hi,verd))
print("\nwrote nb4_direction_curve.csv")
