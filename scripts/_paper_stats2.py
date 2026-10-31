import pandas as pd, numpy as np, json, warnings, collections
warnings.filterwarnings("ignore")
O={}

# SOCIAL POSTS
sp=pd.read_parquet("data/normalized/social/social_posts.parquet",
   columns=["platform","language","created_at","text_length","subreddit_or_channel","query_type","is_duplicate"])
O["social_posts"]={
 "n":int(len(sp)),
 "by_platform":sp["platform"].value_counts(dropna=False).to_dict(),
 "is_duplicate":sp["is_duplicate"].value_counts(dropna=False).to_dict(),
 "top_language":sp["language"].value_counts(dropna=False).head(8).to_dict(),
 "text_length":{k:round(float(v),1) for k,v in sp["text_length"].describe().to_dict().items()},
 "top_channels":sp["subreddit_or_channel"].value_counts().head(20).to_dict(),
 "n_channels":int(sp["subreddit_or_channel"].nunique()),
 "query_type":sp["query_type"].value_counts(dropna=False).head(12).to_dict(),
}
sp["created_at"]=pd.to_datetime(sp["created_at"],utc=True,errors="coerce")
bym=sp.dropna(subset=["created_at"]).groupby(sp["created_at"].dt.to_period("M")).size()
O["social_posts"]["by_month"]={str(k):int(v) for k,v in bym.items()}
del sp

# POST ENRICHMENT
en=pd.read_parquet("data/normalized/social/post_enrichment.parquet",
   columns=["platform","language","sentiment_label","sentiment_compound","sentiment_method","entity_method","mentioned_entities","hashtags"])
O["enrichment"]={
 "n":int(len(en)),
 "sentiment_label":en["sentiment_label"].value_counts(dropna=False).to_dict(),
 "sentiment_compound":{k:round(float(v),4) for k,v in en["sentiment_compound"].describe().to_dict().items()},
 "sentiment_method":en["sentiment_method"].value_counts(dropna=False).to_dict(),
 "entity_method":en["entity_method"].value_counts(dropna=False).to_dict(),
 "top_language":en["language"].value_counts(dropna=False).head(8).to_dict(),
 "sentiment_by_platform":{p:en.loc[en.platform==p,"sentiment_label"].value_counts().to_dict() for p in en["platform"].dropna().unique()},
}
# top entities + hashtags from list columns
def topcounts(series,n=25):
    c=collections.Counter()
    for v in series:
        if v is None: continue
        try:
            it=list(v)
        except: continue
        for x in it:
            x=str(x).strip()
            if x: c[x]+=1
    return dict(c.most_common(n))
O["enrichment"]["top_entities"]=topcounts(en["mentioned_entities"],30)
O["enrichment"]["top_hashtags"]=topcounts(en["hashtags"],25)
del en

# LINKED POSTS
lk=pd.read_parquet("data/normalized/social/linked_posts.parquet",
   columns=["relevance_label","final_relevance_label","confidence","final_confidence","method","final_method","platform","llm_applied","domain_mismatch","constraint_match"])
O["linked"]={
 "n":int(len(lk)),
 "relevance_label":lk["relevance_label"].value_counts(dropna=False).to_dict(),
 "final_relevance_label":lk["final_relevance_label"].value_counts(dropna=False).to_dict(),
 "method":lk["method"].value_counts(dropna=False).head(10).to_dict(),
 "final_confidence":{k:round(float(v),4) for k,v in lk["final_confidence"].describe().to_dict().items()},
 "by_platform":lk["platform"].value_counts(dropna=False).to_dict(),
 "llm_applied":lk["llm_applied"].value_counts(dropna=False).to_dict(),
}
# confidence histogram (0..1 in 0.05 bins)
ch=pd.cut(lk["final_confidence"].clip(0,1),bins=np.arange(0,1.0001,0.05))
O["linked"]["confidence_hist"]={str(k):int(v) for k,v in ch.value_counts().sort_index().items()}
del lk

# STANCE
st=pd.read_parquet("data/normalized/social/stance_labels.parquet",
   columns=["stance_label","method","market_id","used_llm","confidence","target_outcome","stance_score"])
O["stance"]={
 "n":int(len(st)),
 "stance_label":st["stance_label"].value_counts(dropna=False).to_dict(),
 "method":st["method"].value_counts(dropna=False).head(8).to_dict(),
 "used_llm":st["used_llm"].value_counts(dropna=False).to_dict(),
 "n_markets":int(st["market_id"].nunique()),
 "confidence":{k:round(float(v),3) for k,v in st["confidence"].describe().to_dict().items()},
}
del st

# TOPICS
tp=pd.read_parquet("data/normalized/social/topics.parquet")
O["topics"]=tp.to_dict(orient="records")

# SOCIAL PRESSURE
spf=pd.read_parquet("data/analytical/features/social_pressure.parquet")
O["pressure"]={
 "n":int(len(spf)),
 "direction":spf["pressure_direction"].value_counts(dropna=False).to_dict(),
 "spi_shrunk":{k:round(float(v),4) for k,v in spf["spi_score_shrunk"].describe().to_dict().items()},
 "by_domain_accepted":{d:int(spf.loc[spf.domain==d,"accepted_link_count"].sum()) for d in spf["domain"].dropna().unique()},
 "markets_with_links":int((spf["accepted_link_count"]>0).sum()),
 "markets_directional":int(spf["directional_stance_count"].fillna(0).gt(0).sum()),
 "evidence_buckets":{
   "0":int((spf.accepted_link_count==0).sum()),
   "1-9":int(spf.accepted_link_count.between(1,9).sum()),
   "10-99":int(spf.accepted_link_count.between(10,99).sum()),
   "100+":int((spf.accepted_link_count>=100).sum()),
 },
 "supports_yes_total":int(spf["supports_yes_count"].sum()),
 "supports_no_total":int(spf["supports_no_count"].sum()),
}
with open("scripts/_paper_stats_2.json","w") as f: json.dump(O,f,indent=1,default=str)
print(json.dumps(O,indent=1,default=str))
