"""Generate all figures for the PolySignal technical report.
Outputs vector PDFs into pdf_latex/figures/. Read-only over the data; never
modifies the pipeline. Run with the project venv python.
"""
import json, warnings, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter, MaxNLocator
warnings.filterwarnings("ignore")


def _kfmt(x, _pos=None):
    """Compact thousands formatting for axis ticks: 50000 -> '50k'."""
    if abs(x) >= 1000:
        v = x / 1000.0
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return f"{x:.0f}"

FIG = "pdf_latex/figures"
import os; os.makedirs(FIG, exist_ok=True)

# ---- house style -----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.autolayout": False,
})
# muted academic palette
C = {"politics": "#3b6ea5", "finance": "#c46a1f", "sports": "#4f8a4f",
     "blue": "#3b6ea5", "orange": "#c46a1f", "green": "#4f8a4f",
     "red": "#a83232", "grey": "#7a7a7a", "purple": "#6d5b97",
     "yes": "#4f8a4f", "no": "#a83232", "neutral": "#9a9a9a"}

S1 = json.load(open("scripts/_paper_stats_1.json"))
S2 = json.load(open("scripts/_paper_stats_2.json"))
MET = json.load(open("data/analytical/backtests/real_optional/metrics_summary.json"))

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ============================================================ 1 domain + res
def f_domain():
    th = S2 if False else S1["markets"]["by_thematic"]
    order = ["politics", "finance", "sports"]
    rx = S1["markets"]["domain_x_resolution"]  # keyed by fine domain
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax[0].bar(order, [th[k] for k in order],
              color=[C[k] for k in order], width=0.62)
    for i, k in enumerate(order):
        ax[0].text(i, th[k] + 1.5, th[k], ha="center", fontsize=9)
    ax[0].set_ylabel("number of contracts"); ax[0].set_ylim(0, 115)
    ax[0].set_title("(a) Resolved contracts per thematic domain")
    # overall resolution
    yn = S1["markets"]["resolution"]
    ax[1].bar(["NO", "YES"], [yn.get("no", 0), yn.get("yes", 0)],
              color=[C["no"], C["yes"]], width=0.55)
    for i, v in enumerate([yn.get("no", 0), yn.get("yes", 0)]):
        ax[1].text(i, v + 3, v, ha="center", fontsize=9)
    ax[1].set_title("(b) Resolution outcome (all 300)")
    ax[1].set_ylabel("number of contracts"); ax[1].set_ylim(0, 270)
    save(fig, "fig_domain_resolution")

# ============================================================ 2 duration/vol
def f_duration_volume():
    # Self-contained: derive duration + volume straight from the canonical markets
    # table (the old scratch parquet is no longer kept).
    m = pd.read_parquet("data/normalized/polymarket/markets.parquet",
                        columns=["market_id", "thematic_category",
                                 "created_at", "close_time", "volume"])
    m["created_at"] = pd.to_datetime(m["created_at"], errors="coerce", utc=True)
    m["close_time"] = pd.to_datetime(m["close_time"], errors="coerce", utc=True)
    m["duration_days"] = (m["close_time"] - m["created_at"]).dt.total_seconds() / 86400.0
    order = ["politics", "finance", "sports"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    data = [m.loc[m.thematic_category == k, "duration_days"].dropna().values for k in order]
    bp = ax[0].boxplot(data, tick_labels=order, patch_artist=True, widths=0.55,
                       showfliers=False, medianprops=dict(color="black"))
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(C[k]); patch.set_alpha(0.65)
    ax[0].set_ylabel("contract lifetime (days)")
    ax[0].set_title("(a) Contract duration by domain")
    vol = [np.log10(m.loc[m.thematic_category == k, "volume"].clip(lower=1)) for k in order]
    bp2 = ax[1].boxplot(vol, tick_labels=order, patch_artist=True, widths=0.55,
                        showfliers=False, medianprops=dict(color="black"))
    for patch, k in zip(bp2["boxes"], order):
        patch.set_facecolor(C[k]); patch.set_alpha(0.65)
    ax[1].set_ylabel(r"trading volume  ($\log_{10}$ USD)")
    ax[1].set_title("(b) Trading volume by domain")
    save(fig, "fig_duration_volume")

# ============================================================ 3 volatility
def f_volatility():
    vd = S1["price_points"]["volatility_by_domain"]
    # collapse fine -> thematic
    th = {"politics": ["politics"], "finance": ["crypto", "economy"],
          "sports": ["sports", "culture"]}
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    order = ["politics", "finance", "sports"]
    vals = []
    for k in order:
        xs = [vd[d] for d in th[k] if d in vd]
        vals.append(np.mean(xs) if xs else 0)
    ax.bar(order, vals, color=[C[k] for k in order], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("median YES-price std. dev.")
    ax.set_title("Price volatility by domain")
    save(fig, "fig_price_volatility")

# ============================================================ 4 informativeness
def f_informativeness():
    inf = pd.DataFrame(S1["price_informativeness_vs_horizon"])
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax1.plot(inf.horizon_days, inf.auc, "-o", color=C["blue"], label="AUC")
    ax1.set_xlabel("days before resolution (prediction horizon)")
    ax1.set_ylabel("AUC", color=C["blue"]); ax1.set_ylim(0.85, 1.0)
    ax1.tick_params(axis="y", labelcolor=C["blue"])
    ax1.invert_xaxis()
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(inf.horizon_days, inf.brier, "-s", color=C["orange"], label="Brier")
    ax2.set_ylabel("Brier score", color=C["orange"]); ax2.set_ylim(0, 0.12)
    ax2.tick_params(axis="y", labelcolor=C["orange"]); ax2.spines["top"].set_visible(False)
    ax1.set_title("Market-price informativeness vs. horizon")
    save(fig, "fig_price_informativeness")

# ============================================================ 5 timelines
def f_timelines():
    tl = pd.read_parquet("data/analytical/timelines/market_timelines.parquet")
    tl["date"] = pd.to_datetime(tl["date"], errors="coerce")
    # rank markets by informativeness of social line (days * max strength)
    g = tl.groupby("market_id").agg(days=("date", "size"),
        strg=("signal_strength", "max"),
        dom=("domain", "first"), q=("question", "first"),
        res=("resolution", "first")).reset_index()
    g = g[g.days >= 25]
    # preferred curated set spanning domains
    prefer = [
        "0x5412ae25e97078f814157de948459d59c6221b4c4c495fdd57b536543ad36729",  # Russia/Ukraine ceasefire
        "0x36e8ca24d2a13435f519f15580d42b45cdbe3bc425fb39a7c2733260357af0fa",  # Fed no change Apr 2026
        "0x35cc41270f5cdfd59b45e68ab85dc51b0b900d286a6c74ea006b8572d9ff5934",  # Fed -25bps Jan 2026
        "0x773abaa5fe55e5cde51a261f444b7921652a4e059ead6b3be9fe56499c2d4609",  # US x Iran ceasefire Apr 15
    ]
    chosen = [mid for mid in prefer if mid in set(g.market_id)]
    if len(chosen) < 4:
        extra = g.sort_values(["strg", "days"], ascending=False).market_id.tolist()
        for mid in extra:
            if mid not in chosen:
                chosen.append(mid)
            if len(chosen) == 4:
                break
    chosen = chosen[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
    for ax, mid in zip(axes.ravel(), chosen):
        d = tl[tl.market_id == mid].sort_values("date")
        q = str(d["question"].iloc[0]); res = str(d["resolution"].iloc[0]).upper()
        ax.plot(d.date, d.market_prob, color=C["blue"], lw=1.6, label="market P(YES)")
        ss = d.social_signal.where(d.signal_strength > 0.05)
        ax.plot(d.date, ss, color=C["orange"], lw=1.4, ls="--", label="social signal")
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(0.5, color=C["grey"], lw=0.5, ls=":")
        title = (q[:40] + "...") if len(q) > 42 else q
        ax.set_title(f"{title}\n(resolved {res})", fontsize=8.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="x", labelrotation=25, labelsize=6.8)
        ax.tick_params(axis="y", labelsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C["blue"], lw=1.8, label="market P(YES)"),
               Line2D([0], [0], color=C["orange"], lw=1.6, ls="--", label="social signal")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Polymarket implied probability vs. smoothed social signal", y=0.995)
    fig.subplots_adjust(hspace=0.72, wspace=0.2, top=0.9, bottom=0.12)
    save(fig, "fig_example_timelines")

# ====================================================== 5b lead-lag / correlation
def f_leadlag():
    """Clean replacement for the noisy example-timelines: (a) lead-lag
    cross-correlation showing no positive-lead peak, (b) the distribution of
    per-contract social-vs-price correlations. Both computed live from the timelines."""
    tl = pd.read_parquet("data/analytical/timelines/market_timelines.parquet")
    d = tl.dropna(subset=["social_signal", "market_prob"])
    m = d[d["signal_strength"] > 0.2]
    well = m["market_id"].value_counts()
    well = well[well >= 20].index
    lags = list(range(-7, 8))
    rk = []
    for k in lags:
        xs = []
        for mid in well:
            g = tl[tl.market_id == mid].sort_values("date")
            ds = g["social_signal"].diff()
            dp = g["market_prob"].diff().shift(-k)  # k>0 => price moves after social
            v = pd.concat([ds, dp], axis=1).dropna()
            if len(v) > 3:
                xs.append(v)
        a = pd.concat(xs)
        rk.append(float(a.iloc[:, 0].corr(a.iloc[:, 1])))
    pc = []
    for mid, g in m.groupby("market_id"):
        if len(g) >= 5 and g["social_signal"].std() > 0 and g["market_prob"].std() > 0:
            pc.append(g["social_signal"].corr(g["market_prob"]))
    pc = pd.Series(pc).dropna()

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
    cols = [C["green"] if k > 0 else (C["grey"] if k == 0 else C["orange"]) for k in lags]
    ax[0].axhline(0, color="black", lw=0.7)
    ax[0].axvline(0, color=C["grey"], lw=0.6, ls=":")
    ax[0].bar(lags, rk, color=cols, width=0.72, alpha=0.9)
    ax[0].set_ylim(-0.06, 0.06)
    ax[0].set_xticks([-6, -3, 0, 3, 6])
    ax[0].set_xlabel(r"lag $k$ (days)")
    ax[0].set_ylabel(r"corr($\Delta$social$_t,\,\Delta$price$_{t+k}$)")
    ax[0].set_title("(a) Lead--lag cross-correlation")
    ax[0].annotate("social leads", xy=(4.2, 0.052), color=C["green"], fontsize=7.6, ha="center")
    ax[0].annotate("social lags", xy=(-4.2, 0.052), color=C["orange"], fontsize=7.6, ha="center")

    ax[1].hist(pc, bins=15, color=C["blue"], alpha=0.8)
    ax[1].axvline(0, color="black", lw=0.7, ls=":")
    med = float(pc.median())
    ax[1].axvline(med, color=C["red"], lw=1.3, ls="--")
    ax[1].text(med + 0.04, ax[1].get_ylim()[1] * 0.82, f"median {med:.2f}",
               color=C["red"], fontsize=8)
    ax[1].set_xlim(-1, 1)
    ax[1].set_xlabel("per-contract corr (social vs. price)")
    ax[1].set_ylabel("contracts")
    ax[1].set_title("(b) Weak, mostly-zero correlation")
    fig.subplots_adjust(wspace=0.32, bottom=0.18)
    save(fig, "fig_leadlag")

# ============================================================ 6 social volume
def f_social_volume():
    bym = S2["social_posts"]["by_month"]
    s = pd.Series({pd.Period(k): v for k, v in bym.items()}).sort_index()
    s = s[s.index >= pd.Period("2018-01")]
    x = s.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.fill_between(x, s.values, color=C["blue"], alpha=0.35)
    ax.plot(x, s.values, color=C["blue"], lw=1.0)
    ax.axvspan(pd.Timestamp("2024-06-01"), x.max(), color=C["orange"], alpha=0.10)
    ax.text(pd.Timestamp("2024-07-01"), s.max() * 0.9,
            "post-2024 analysis window", fontsize=8, color=C["orange"])
    ax.set_ylabel("posts per month"); ax.set_title("Social corpus volume over time (monthly)")
    save(fig, "fig_social_volume")

# ============================================================ 7 platform/channel
def f_channels():
    plat = S2["social_posts"]["by_platform"]
    ch = S2["social_posts"]["top_channels"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4),
                           gridspec_kw={"width_ratios": [1, 1.5]})
    # all three collected platforms, in fixed order, each with its own colour
    pcol = {"telegram": C["blue"], "reddit": C["orange"], "twitter": C["purple"]}
    pk = [k for k in ["telegram", "reddit", "twitter"] if k in plat]
    pk += [k for k in plat if k not in pk]
    pv = [plat[k] for k in pk]
    ax[0].bar(pk, pv, color=[pcol.get(k, C["grey"]) for k in pk], width=0.62)
    ax[0].set_yscale("log"); ax[0].set_ylabel("posts (log scale)")
    for i, v in enumerate(pv):
        ax[0].text(i, v * 1.25, f"{v:,}", ha="center", fontsize=8)
    ax[0].set_ylim(top=max(pv) * 4)
    ax[0].tick_params(axis="x", labelsize=8.5)
    ax[0].set_title("(a) Posts per platform")
    items = list(ch.items())[:12][::-1]
    names = [k for k, _ in items]; vals = [v for _, v in items]
    ax[1].barh(names, vals, color=C["purple"], alpha=0.8)
    ax[1].set_xlabel("posts"); ax[1].set_title("(b) Top Telegram channels by volume")
    ax[1].tick_params(axis="y", labelsize=7.5)
    ax[1].xaxis.set_major_locator(MaxNLocator(5))
    ax[1].xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax[1].tick_params(axis="x", labelsize=8)
    ax[1].margins(x=0.10)
    save(fig, "fig_corpus_platform_channels")

# ============================================================ 8 sentiment
def f_sentiment():
    sl = S2["enrichment"]["sentiment_label"]
    order = ["negative", "neutral", "positive"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    cols = {"negative": C["no"], "neutral": C["neutral"], "positive": C["yes"]}
    tot = sum(sl.values())
    ax[0].bar(order, [sl[k] for k in order], color=[cols[k] for k in order], width=0.6)
    for i, k in enumerate(order):
        ax[0].text(i, sl[k] + tot*0.01, f"{sl[k]/tot*100:.0f}%", ha="center", fontsize=9)
    ax[0].set_ylabel("posts"); ax[0].set_title("(a) Sentiment label distribution")
    # compound dist approximation from describe -> draw normal-ish via quantiles
    sc = S2["enrichment"]["sentiment_compound"]
    # rough histogram from real data
    en = pd.read_parquet("data/normalized/social/post_enrichment.parquet",
                         columns=["sentiment_compound"])
    ax[1].hist(en.sentiment_compound.dropna(), bins=40, color=C["blue"], alpha=0.75)
    ax[1].axvline(0, color="black", lw=0.6, ls=":")
    ax[1].set_xlabel("VADER compound score"); ax[1].set_ylabel("posts")
    ax[1].set_title(f"(b) Compound distribution (mean {sc['mean']:+.3f})")
    save(fig, "fig_sentiment")

# ============================================================ 9 entities
def f_entities():
    ents = S2["enrichment"]["top_entities"]
    # keep informative entities (drop obvious channel/app/boilerplate noise)
    noise = {"FirstSquaw", "WalterBloomberg", "Telegram", "WhatsApp", "Viber",
             "YouTube", "JUST IN", "BREAKING", "Markets"}
    pairs = []
    for k, v in ents.items():
        d = json.loads(k.replace("'", '"')) if isinstance(k, str) and k.startswith("{") else None
        name = d["text"] if d else str(k)
        if name in noise:
            continue
        pairs.append((name, v))
    pairs = pairs[:15][::-1]
    names = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.barh(names, vals, color=C["blue"], alpha=0.8)
    ax.set_xlabel("mentions"); ax.set_title("Most frequent named entities")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.tick_params(axis="x", labelsize=8)
    ax.margins(x=0.10)
    save(fig, "fig_entities")

# ============================================================ 10 topics
def f_topics():
    tp = pd.DataFrame(S2["topics"]).sort_values("weight")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(tp.label, tp.weight, color=C["purple"], alpha=0.8)
    ax.set_xlabel("topic mass (share of linked discourse)")
    ax.set_title("LDA topics over linked posts (top terms)")
    ax.tick_params(axis="y", labelsize=7.8)
    save(fig, "fig_topics")

# ============================================================ 11 linking funnel
def f_funnel():
    # Derived live from the refreshed stats so the funnel always matches the data.
    rl = S2["linked"]["relevance_label"]
    corpus = S2["social_posts"]["n"]
    candidates = S2["linked"]["n"]
    relweak = rl.get("relevant", 0) + rl.get("weakly_relevant", 0)
    accepted = int(sum(S2["pressure"]["by_domain_accepted"].values()))
    stages = [("deduplicated\nposts", corpus),
              ("candidate\n(post, contract)\npairs", candidates),
              ("relevant +\nweak relevant", relweak),
              ("accepted\nevidence", accepted)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = [s[0] for s in stages]; vals = [s[1] for s in stages]
    cols = [C["grey"], C["blue"], C["blue"], C["green"]]
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.85, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.3, f"{v:,}", ha="center", fontsize=8.5)
    ax.set_ylabel("rows (log scale)")
    ax.set_title("Contract--post linking funnel (relevance scoring fans out, then narrows)")
    ax.set_ylim(1e4, max(vals) * 4)
    save(fig, "fig_linking_funnel")

# ============================================================ 12 link confidence
def f_confidence():
    ch = S2["linked"]["confidence_hist"]
    centers = []; vals = []
    for k, v in ch.items():
        lo = float(k.split(",")[0].strip("("))
        hi = float(k.split(",")[1].strip(" ]"))
        centers.append((lo + hi) / 2); vals.append(v)
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.bar(centers, vals, width=0.045, color=C["blue"], alpha=0.8)
    ax.axvline(0.6, color=C["red"], lw=1.4, ls="--")
    ax.text(0.61, max(vals) * 0.85, "acceptance\nthreshold 0.60",
            color=C["red"], fontsize=8.5)
    ax.set_xlabel("final link confidence"); ax.set_ylabel("candidate pairs")
    ax.set_title("Distribution of link-confidence scores")
    save(fig, "fig_link_confidence")

# ============================================================ 13 stance
def f_stance():
    sl = S2["stance"]["stance_label"]
    order = ["supports_yes", "supports_no", "neutral", "ambiguous", "insufficient_context"]
    lbl = {"supports_yes": "supports\nYES", "supports_no": "supports\nNO",
           "neutral": "neutral", "ambiguous": "ambiguous",
           "insufficient_context": "insufficient\ncontext"}
    cols = {"supports_yes": C["yes"], "supports_no": C["no"], "neutral": C["neutral"],
            "ambiguous": C["grey"], "insufficient_context": "#c9c9c9"}
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    vals = [sl.get(k, 0) for k in order]
    ax[0].bar([lbl[k] for k in order], vals, color=[cols[k] for k in order], width=0.62)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 300, f"{v:,}", ha="center", fontsize=8)
    ax[0].set_ylabel("post-outcome rows"); ax[0].set_title("(a) Stance label distribution")
    ax[0].tick_params(axis="x", labelsize=8)
    # directional bias vs actual resolution
    sy, sn = S2["pressure"]["supports_yes_total"], S2["pressure"]["supports_no_total"]
    ry, rn = S1["markets"]["resolution"]["yes"], S1["markets"]["resolution"]["no"]
    x = np.arange(2); w = 0.36
    ax[1].bar(x - w/2, [sy/(sy+sn), sn/(sy+sn)], w, label="social stance",
              color=C["purple"], alpha=0.85)
    ax[1].bar(x + w/2, [ry/(ry+rn), rn/(ry+rn)], w, label="actual outcome",
              color=C["grey"], alpha=0.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["YES", "NO"])
    ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    ax[1].set_title("(b) Directional bias vs.\nactual resolution")
    ax[1].legend(fontsize=7.5)
    save(fig, "fig_stance")

# ============================================================ 14 evidence coverage
def f_coverage():
    bd = S2["pressure"]["by_domain_accepted"]
    th = {"politics": ["politics"], "finance": ["crypto", "economy"],
          "sports": ["sports", "culture"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    order = ["politics", "finance", "sports"]
    vals = [sum(bd.get(d, 0) for d in th[k]) for k in order]
    ax[0].bar(order, vals, color=[C[k] for k in order], width=0.6)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 400, f"{v:,}", ha="center", fontsize=9)
    ax[0].set_ylabel("accepted evidence links")
    ax[0].set_title("(a) Accepted evidence by domain")
    eb = S2["pressure"]["evidence_buckets"]
    ks = ["0", "1-9", "10-99", "100+"]
    ax[1].bar(ks, [eb[k] for k in ks], color=C["blue"], alpha=0.8, width=0.6)
    for i, k in enumerate(ks):
        ax[1].text(i, eb[k] + 3, eb[k], ha="center", fontsize=9)
    ax[1].set_xlabel("accepted links per contract")
    ax[1].set_ylabel("number of contracts")
    ax[1].set_title("(b) Evidence depth per contract")
    save(fig, "fig_evidence_coverage")

# ============================================================ 15 model comparison
def f_models():
    mm = MET["models"]
    order = ["constant", "market_only", "social_only", "market_social"]
    nice = {"constant": "base rate", "market_only": "market\nonly",
            "social_only": "social\nonly", "market_social": "market\n+ social"}
    cols = {"constant": C["grey"], "market_only": C["blue"],
            "social_only": C["orange"], "market_social": C["green"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    brier = [mm[k]["brier_score"] for k in order]
    ax[0].bar([nice[k] for k in order], brier, color=[cols[k] for k in order], width=0.62)
    for i, v in enumerate(brier):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
    ax[0].set_ylabel("Brier score  (lower better)")
    ax[0].set_title("(a) Brier score"); ax[0].set_ylim(0, 0.19)
    auc = [mm[k]["auc"] for k in order]
    ax[1].bar([nice[k] for k in order], auc, color=[cols[k] for k in order], width=0.62)
    ax[1].axhline(0.5, color=C["red"], lw=0.8, ls=":")
    for i, v in enumerate(auc):
        ax[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax[1].set_ylabel("AUC  (higher better)")
    ax[1].set_title("(b) AUC-ROC"); ax[1].set_ylim(0, 1.08)
    save(fig, "fig_model_comparison")

# ============================================================ 16 calibration
def f_calibration():
    cb = pd.read_csv("data/analytical/backtests/real_optional/calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":", label="perfect")
    for mdl, col, mk in [("market_only", C["blue"], "o"),
                          ("market_social", C["green"], "s")]:
        d = cb[(cb.model == mdl) & (cb.scope == "available_rows")].sort_values("mean_probability")
        ax.plot(d.mean_probability, d.observed_rate, mk + "-", color=col,
                label=mdl.replace("_", " "), ms=5, lw=1.3)
    ax.set_xlabel("mean predicted P(YES)"); ax.set_ylabel("observed YES rate")
    ax.set_title("Reliability diagram"); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_calibration")

# ============================================================ 17 feature importance
def f_features():
    fi = pd.read_csv("data/analytical/backtests/real_optional/feature_importance.csv")
    d = fi[fi.model == "market_social"].copy()
    d["feat"] = d.feature.str.replace("_", " ")
    d = d.sort_values("coefficient")
    cols = [C["green"] if c > 0 else C["red"] for c in d.coefficient]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(d.feat, d.coefficient, color=cols, alpha=0.85)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("logistic-regression coefficient (standardized)")
    ax.set_title("Feature importance: market+social model")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "fig_feature_importance")

# ====================================================== 17b task-3 temporal CV
def f_task3():
    t3 = json.load(open("data/analytical/backtests/real_optional/task3_results.json"))
    models = t3["models"]
    order = ["logistic_regression", "linear_svm", "naive_bayes", "gradient_boosting"]
    nice = {"logistic_regression": "logistic", "linear_svm": "linear SVM",
            "naive_bayes": "naive Bayes", "gradient_boosting": "grad. boost"}
    ab_order = ["market_only", "social_only", "market_social"]
    ab_col = {"market_only": C["blue"], "social_only": C["orange"],
              "market_social": C["green"]}
    ab_nice = {"market_only": "market only", "social_only": "social only",
               "market_social": "market + social"}
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for metric, axi, title, ylim in [("macro_f1", ax[0], "(a) macro-F1", 1.05),
                                      ("auc", ax[1], "(b) AUC-ROC", 1.08)]:
        for j, ab in enumerate(ab_order):
            vals = [models[m][ab].get(metric) or 0.0 for m in order]
            axi.bar(x + (j - 1) * w, vals, w, color=ab_col[ab], label=ab_nice[ab])
        const = t3["baseline_constant"].get(metric) or 0.0
        axi.axhline(const, color=C["grey"], lw=0.9, ls="--", label="constant")
        if metric == "auc":
            axi.axhline(0.5, color=C["red"], lw=0.8, ls=":")
        axi.set_xticks(x); axi.set_xticklabels([nice[m] for m in order], fontsize=8, rotation=12)
        axi.set_title(title); axi.set_ylim(0, ylim)
    ax[0].set_ylabel("score (out-of-fold, temporal CV)")
    ax[0].legend(fontsize=7.5, ncol=2, loc="lower center")
    save(fig, "fig_task3_models")

# ====================================================== 18b cross-platform
def f_cross_platform():
    summ = pd.read_parquet("data/analytical/analytics/cross_platform_summary.parquet")
    comp = pd.read_parquet("data/analytical/analytics/cross_platform_comparison.parquet")
    order = ["politics", "finance", "sports"]
    summ = summ.set_index("domain").reindex(order)
    single = summ["single_platform"].fillna(0).values
    multi = summ["multi_platform"].fillna(0).values
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax[0].bar(order, single, color=C["grey"], width=0.6, label="single-platform")
    ax[0].bar(order, multi, bottom=single, color=C["blue"], width=0.6, label="multi-platform")
    totals = single + multi
    ax[0].set_ylim(0, max(totals) * 1.30)
    for i, (s, m) in enumerate(zip(single, multi)):
        ax[0].text(i, s + m + max(totals) * 0.02, f"{int(s + m)}", ha="center", fontsize=8.5)
    ax[0].set_ylabel("contracts with social discourse")
    ax[0].set_title("(a) Platform coverage by domain")
    ax[0].legend(fontsize=7.5, loc="upper center", ncol=2, columnspacing=1.0,
                 handlelength=1.3, borderaxespad=0.2)
    md = comp[comp["n_platforms"] >= 2]
    ax[1].hist(md["dominant_share"].dropna(), bins=20, color=C["purple"], alpha=0.8)
    med = float(md["dominant_share"].median())
    ax[1].axvline(med, color=C["red"], lw=1.3, ls="--")
    ax[1].text(med - 0.02, ax[1].get_ylim()[1] * 0.8, f"median {med:.2f}",
               color=C["red"], fontsize=8.5, ha="right")
    ax[1].set_xlabel("dominant-platform share")
    ax[1].set_ylabel("contracts")
    ax[1].set_title("(b) One platform dominates the mix")
    save(fig, "fig_cross_platform")

# ============================================================ 18 hype mismatch
def f_hype():
    diag = json.load(open("data/analytical/backtests/real_optional/backtest_report.json"))
    pv = diag["social_signal_diagnostics"]["pressure_direction_vs_target"]
    # among markets where social pressure says YES, how do they actually resolve?
    yes = pv.get("YES", {});
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    cats = ["pressure = YES", "pressure = NO", "insufficient"]
    yy = [pv.get("YES", {}).get("target_yes", 0), pv.get("NO", {}).get("target_yes", 0),
          pv.get("INSUFFICIENT_EVIDENCE", {}).get("target_yes", 0)]
    nn = [pv.get("YES", {}).get("target_no", 0), pv.get("NO", {}).get("target_no", 0),
          pv.get("INSUFFICIENT_EVIDENCE", {}).get("target_no", 0)]
    x = np.arange(3)
    ax.bar(x, yy, color=C["yes"], label="resolved YES")
    ax.bar(x, nn, bottom=yy, color=C["no"], label="resolved NO")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("number of contracts")
    ax.set_title("Social pressure direction vs. actual outcome")
    ax.legend(fontsize=8)
    save(fig, "fig_hype_mismatch")

for fn in [f_domain, f_duration_volume, f_volatility, f_informativeness,
           f_leadlag, f_social_volume, f_channels, f_sentiment, f_entities,
           f_topics, f_funnel, f_confidence, f_stance, f_coverage, f_models,
           f_calibration, f_features, f_task3, f_cross_platform, f_hype]:
    try:
        fn()
    except Exception as e:
        import traceback; print("FAIL", fn.__name__, e); traceback.print_exc()
print("\nDONE")
