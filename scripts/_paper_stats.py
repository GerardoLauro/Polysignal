import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
OUT={}

# ---------- MARKETS ----------
m=pd.read_parquet("data/normalized/polymarket/markets.parquet")
m["created_at"]=pd.to_datetime(m["created_at"],utc=True,errors="coerce")
m["close_time"]=pd.to_datetime(m["close_time"],utc=True,errors="coerce")
m["duration_days"]=(m["close_time"]-m["created_at"]).dt.total_seconds()/86400
OUT["markets"]={
 "n":int(len(m)),
 "by_domain":m["domain"].value_counts(dropna=False).to_dict(),
 "by_thematic":m["thematic_category"].value_counts(dropna=False).to_dict(),
 "resolution":m["resolution"].value_counts(dropna=False).to_dict(),
 "is_resolved":m["is_resolved"].value_counts(dropna=False).to_dict(),
 "duration_days":{k:round(float(v),2) for k,v in m["duration_days"].describe().to_dict().items()},
 "duration_by_domain":{d:{k:round(float(v),1) for k,v in m.loc[m.domain==d,"duration_days"].describe().to_dict().items()} for d in m["domain"].dropna().unique()},
 "volume":{k:round(float(v),1) for k,v in m["volume"].describe().to_dict().items()},
 "volume_by_domain":{d:round(float(m.loc[m.domain==d,"volume"].median()),1) for d in m["domain"].dropna().unique()},
 "liquidity":{k:round(float(v),1) for k,v in m["liquidity"].describe().to_dict().items()},
}
# resolution YES/NO mapping
res=m["resolution"].astype(str).str.upper()
OUT["markets"]["yes_no"]={"YES":int(res.str.contains("YES").sum()),"NO":int(res.str.contains("NO").sum())}
# domain x resolution
ct=pd.crosstab(m["domain"],res)
OUT["markets"]["domain_x_resolution"]=ct.to_dict()

# ---------- OUTCOMES + price volatility ----------
pp=pd.read_parquet("data/normalized/polymarket/price_points.parquet",columns=["market_id","outcome_id","timestamp","price"])
oc=pd.read_parquet("data/normalized/polymarket/outcomes.parquet")
yes_ids=set(oc.loc[oc["label"].astype(str).str.upper().str.contains("YES"),"outcome_id"])
pp["timestamp"]=pd.to_datetime(pp["timestamp"],utc=True,errors="coerce")
ppy=pp[pp["outcome_id"].isin(yes_ids)].copy()
OUT["price_points"]={
 "n_total":int(len(pp)),
 "n_markets":int(pp["market_id"].nunique()),
 "per_market":{k:round(float(v),1) for k,v in pp.groupby("market_id").size().describe().to_dict().items()},
}
vol=ppy.groupby("market_id")["price"].std()
OUT["price_points"]["yes_price_volatility"]={k:round(float(v),4) for k,v in vol.describe().to_dict().items()}
# volatility by domain
dmap=m.set_index("market_id")["domain"].to_dict()
voldf=vol.reset_index(); voldf["domain"]=voldf["market_id"].map(dmap)
OUT["price_points"]["volatility_by_domain"]={d:round(float(voldf.loc[voldf.domain==d,"price"].median()),4) for d in voldf["domain"].dropna().unique()}

# ---------- PRICE INFORMATIVENESS vs HORIZON (Task 2.1) ----------
# resolved YES label per market
res_yes=res.str.contains("YES").astype(int); res_yes.index=m["market_id"]
ppy=ppy.merge(m[["market_id","close_time"]],on="market_id",how="left")
ppy["days_to_close"]=(ppy["close_time"]-ppy["timestamp"]).dt.total_seconds()/86400
horizons=[0,1,3,7,14,30,60,90]
info=[]
for h in horizons:
    # latest price at >= h days before close
    sub=ppy[ppy["days_to_close"]>=h]
    if len(sub)==0: continue
    idx=sub.groupby("market_id")["days_to_close"].idxmin()
    snap=sub.loc[idx,["market_id","price"]].set_index("market_id")
    snap["y"]=snap.index.map(res_yes.to_dict())
    snap=snap.dropna()
    # AUC + brier of price as P(YES)
    from math import isnan
    y=snap["y"].values; p=snap["price"].clip(0,1).values
    brier=float(np.mean((p-y)**2))
    # AUC
    pos=p[y==1]; neg=p[y==0]
    if len(pos)>0 and len(neg)>0:
        # rank-based AUC
        allv=np.concatenate([pos,neg]); order=allv.argsort()
        ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
        auc=(ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    else: auc=float("nan")
    acc=float(np.mean((p>=0.5).astype(int)==y))
    info.append({"horizon_days":h,"n":int(len(snap)),"auc":round(float(auc),4),"brier":round(brier,4),"accuracy":round(acc,4)})
OUT["price_informativeness_vs_horizon"]=info

print(json.dumps(OUT,indent=1,default=str))
with open("scripts/_paper_stats_1.json","w") as f: json.dump(OUT,f,indent=1,default=str)
print("\nSAVED scripts/_paper_stats_1.json")
