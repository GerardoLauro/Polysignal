"""Main enriched social dataset augmentation for UI stress testing.

The generated rows are deterministic and intentionally stored as ordinary
``SocialPostRecord`` rows so they flow through linking, stance, SPI, Branch 09,
and Branch 10 without a parallel demo profile.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from polysignal.domain import Market, Outcome, SocialPostRecord
from polysignal.domain.enums import Platform
from polysignal.storage.parquet_store import (
    dataframe_to_models,
    models_to_dataframe,
    read_parquet,
    write_parquet,
)
from polysignal.storage.paths import DataPaths, get_data_paths

DEFAULT_AUGMENTATION_VERSION = "ui_stress_test_12E"
DEFAULT_MARKETS_PER_RUN = 6
DEFAULT_MESSAGES_PER_MARKET = 150
DEFAULT_SEED = 1205

MARKER_PREFIX = "augmentation_version="


@dataclass(frozen=True)
class MarketAnchor:
    domain_label: str
    market_id: str
    question: str
    close_time: datetime


@dataclass(frozen=True)
class AugmentationPlan:
    markets: list[MarketAnchor]
    generated_rows: list[SocialPostRecord]
    stance_breakdown: dict[str, int]
    platform_breakdown: dict[str, int]
    files_to_update: list[Path]
    existing_rows: int
    previous_augmented_rows: int
    final_rows: int
    dry_run: bool


SELECTED_MARKETS: tuple[tuple[str, str], ...] = (
    ("crypto/finance", "0x6874f3bbffd0ff156b09768d1e1a5bdfaac48a7aaee9fbe1e883dccdc5f3d68d"),
    ("politics/elections", "0x2dae146105569fc37e2c7365f540fd7028b3c1a42fcc1fca7f0dbd9902624e15"),
    ("sports", "0x0851b31ba4f1c35e13c1ca100be710a6cc68bb5c958c37a967e65f03dd828beb"),
    ("technology/social media", "0x77825ad3698fb528b217de4da17a61660be6a1678425679f017a9cf4d7a4ddb2"),
    ("entertainment/culture", "0x0f8f5962a2f9b62abe4d6d88de65a6327a44b749fbecd89a780fcd463824b7f1"),
    ("geopolitics/news fallback", "0xb943884a1226dad793d08f5143b8c13db715a1e40f846ce80e06499654dfcb1e"),
)

_REDDIT_SOURCES = {
    "crypto/finance": ("CryptoCurrency", "Bitcoin", "investing", "stocks"),
    "politics/elections": ("politics", "CanadaPolitics", "worldnews"),
    "sports": ("nba", "MkeBucks", "sports"),
    "technology/social media": ("technology", "elonmusk", "Twitter"),
    "entertainment/culture": ("movies", "boxoffice", "A24"),
    "geopolitics/news fallback": ("worldnews", "Africa", "geopolitics"),
}

_TELEGRAM_SOURCES = {
    "crypto/finance": ("cointelegraph", "CTMarkets", "WatcherGuru"),
    "politics/elections": ("DeItaone", "disclosetv", "politicoalerts"),
    "sports": ("FabrizioRomano", "sportsalerts", "NBABulletin"),
    "technology/social media": ("WatcherGuru", "Techmeme", "DeItaone"),
    "entertainment/culture": ("DeadlineHollywood", "BoxOfficeReport", "VarietyAlerts"),
    "geopolitics/news fallback": ("disclosetv", "DeItaone", "geopoliticslive"),
}

_ANCHORS = {
    "crypto/finance": ("Bitcoin", "BTC", "Q4", "all time high"),
    "politics/elections": ("Canada", "popular vote", "election", "another party"),
    "sports": ("Milwaukee Bucks", "NBA Playoffs", "Bucks", "playoffs"),
    "technology/social media": ("Elon Musk", "tweets", "X", "December"),
    "entertainment/culture": ("Bring Her Back", "opening weekend", "box office", "$3.5-4.5m"),
    "geopolitics/news fallback": (
        "National Congress for Liberty",
        "Burundi",
        "parliamentary election",
        "most seats",
    ),
}

_YES_TEMPLATES = (
    "{anchor0} looks on track to make this YES outcome happen before expiry. {anchor1} signals are stronger than the market is pricing.",
    "I think YES is likely here: {anchor0} momentum keeps improving and the {anchor2} setup is moving in the right direction.",
    "{anchor0} is set to clear the bar if the latest {anchor1} trend holds. YES looks realistic before the deadline.",
    "The path to YES is getting cleaner for {anchor0}. Traders expect this to happen because {anchor2} keeps lining up.",
    "This market is underestimating YES. {anchor0} has enough current momentum and could hit the required outcome in time.",
)

_NO_TEMPLATES = (
    "I do not think {anchor0} gets there. The {anchor1} setup looks weak and this YES outcome is unlikely before expiry.",
    "NO still looks better to me: {anchor0} has too many blockers and will not clear this market's condition.",
    "{anchor0} momentum is fading. This looks off track, and I expect the YES side to fall short before the deadline.",
    "The market is too optimistic on {anchor0}. Recent {anchor2} signals make the YES outcome unlikely.",
    "I would fade YES here. {anchor0} probably misses the deadline and the current trend points to NO.",
)

_NEUTRAL_TEMPLATES = (
    "{anchor0} discussion is active today, with people watching {anchor1} and {anchor2} as the deadline approaches.",
    "Useful context on {anchor0}: the market depends heavily on {anchor1}, but there is no clear directional read yet.",
    "Several updates mention {anchor0} and {anchor2}. I am tracking the data before taking a side on YES or NO.",
    "The {anchor0} market is getting more attention as close approaches. Liquidity and headlines around {anchor1} matter here.",
)

_MIXED_TEMPLATES = (
    "{anchor0} could still go either way. YES has a plausible path, but {anchor1} uncertainty makes NO possible too.",
    "I see arguments on both sides for {anchor0}: some signals look likely, but the timeline may be too tight.",
    "Mixed read on {anchor0}. It might happen if {anchor1} improves, but right now the setup remains uncertain.",
    "There is a YES case for {anchor0}, but NO is not dead because {anchor2} still looks unresolved.",
)


def augmentation_marker(version: str) -> str:
    return f"{MARKER_PREFIX}{version}"


def load_selected_markets(
    paths: DataPaths,
    *,
    markets: int = DEFAULT_MARKETS_PER_RUN,
) -> list[tuple[str, Market]]:
    """Load and validate the fixed market anchors from the existing dataset."""
    if markets != DEFAULT_MARKETS_PER_RUN:
        raise ValueError("this augmentation is pinned to exactly 6 selected markets")
    loaded = {market.market_id: market for market in read_parquet(paths.polymarket_markets_parquet, Market)}
    outcomes_by_market: dict[str, set[str]] = {}
    for outcome in read_parquet(paths.polymarket_outcomes_parquet, Outcome):
        outcomes_by_market.setdefault(outcome.market_id, set()).add(outcome.label.lower())

    selected: list[tuple[str, Market]] = []
    missing: list[str] = []
    for domain_label, market_id in SELECTED_MARKETS:
        market = loaded.get(market_id)
        if market is None:
            missing.append(market_id)
            continue
        if market.close_time is None:
            raise ValueError(f"selected market has no close_time: {market_id}")
        labels = outcomes_by_market.get(market_id, set())
        if not {"yes", "no"}.issubset(labels):
            raise ValueError(f"selected market is not binary YES/NO: {market_id}")
        selected.append((domain_label, market))
    if missing:
        raise ValueError(f"selected market(s) missing from saved Polymarket data: {missing}")
    return selected


def generate_augmented_rows(
    selected: list[tuple[str, Market]],
    *,
    messages_per_market: int = DEFAULT_MESSAGES_PER_MARKET,
    augmentation_version: str = DEFAULT_AUGMENTATION_VERSION,
    seed: int = DEFAULT_SEED,
) -> list[SocialPostRecord]:
    if messages_per_market < 20:
        raise ValueError("messages_per_market must be >= 20 for a realistic stance mix")
    rows: list[SocialPostRecord] = []
    collected_at = datetime.now(tz=timezone.utc)
    for market_index, (domain_label, market) in enumerate(selected):
        rng = random.Random(f"{seed}:{market.market_id}:{messages_per_market}")
        stances = _realistic_stance_sequence(messages_per_market, rng)
        rng.shuffle(stances)
        for index, stance in enumerate(stances):
            platform = Platform.REDDIT if rng.random() < _reddit_share(domain_label) else Platform.TELEGRAM
            created_at = _sample_created_at(market, rng)
            rows.append(
                _build_row(
                    domain_label=domain_label,
                    market=market,
                    platform=platform,
                    stance=stance,
                    index=index,
                    market_index=market_index,
                    created_at=created_at,
                    collected_at=collected_at,
                    augmentation_version=augmentation_version,
                    rng=rng,
                )
            )
    return rows


def plan_augmentation(
    *,
    root: Path = Path("data_social_enriched_eval"),
    markets: int = DEFAULT_MARKETS_PER_RUN,
    messages_per_market: int = DEFAULT_MESSAGES_PER_MARKET,
    augmentation_version: str = DEFAULT_AUGMENTATION_VERSION,
    seed: int = DEFAULT_SEED,
    dry_run: bool = True,
) -> AugmentationPlan:
    paths = get_data_paths(root=root)
    selected = load_selected_markets(paths, markets=markets)
    generated = generate_augmented_rows(
        selected,
        messages_per_market=messages_per_market,
        augmentation_version=augmentation_version,
        seed=seed,
    )
    existing = _read_existing_social_records(paths)
    marker = augmentation_marker(augmentation_version)
    previous_augmented = [row for row in existing if marker in (row.query_provenance or "")]
    final_rows = _merge_generated(existing, generated, augmentation_version=augmentation_version)
    return AugmentationPlan(
        markets=[
            MarketAnchor(
                domain_label=domain_label,
                market_id=market.market_id,
                question=market.question,
                close_time=_as_utc(market.close_time),
            )
            for domain_label, market in selected
        ],
        generated_rows=generated,
        stance_breakdown=dict(Counter(row.retrieval_window_label or "unknown" for row in generated)),
        platform_breakdown=dict(Counter(row.platform.value for row in generated)),
        files_to_update=[paths.social_posts_parquet, paths.social_posts_parquet.with_suffix(".csv")],
        existing_rows=len(existing),
        previous_augmented_rows=len(previous_augmented),
        final_rows=len(final_rows),
        dry_run=dry_run,
    )


def augment_main_social_dataset(
    *,
    root: Path = Path("data_social_enriched_eval"),
    markets: int = DEFAULT_MARKETS_PER_RUN,
    messages_per_market: int = DEFAULT_MESSAGES_PER_MARKET,
    augmentation_version: str = DEFAULT_AUGMENTATION_VERSION,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
) -> AugmentationPlan:
    plan = plan_augmentation(
        root=root,
        markets=markets,
        messages_per_market=messages_per_market,
        augmentation_version=augmentation_version,
        seed=seed,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    paths = get_data_paths(root=root)
    existing = _read_existing_social_records(paths)
    final_rows = _merge_generated(existing, plan.generated_rows, augmentation_version=augmentation_version)
    write_parquet(final_rows, paths.social_posts_parquet)
    csv_path = paths.social_posts_parquet.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    models_to_dataframe(final_rows).to_csv(csv_path, index=False)
    return plan


def _realistic_stance_sequence(total: int, rng: random.Random) -> list[str]:
    neutral = max(12, min(total // 5, int(total * rng.uniform(0.12, 0.19))))
    mixed = max(5, min(total // 10, int(total * rng.uniform(0.04, 0.08))))
    directional = total - neutral - mixed
    yes = int(directional * rng.uniform(0.45, 0.56))
    no = directional - yes
    if yes == no:
        yes += 1
        no -= 1
    return ["supports_yes"] * yes + ["supports_no"] * no + ["neutral"] * neutral + ["mixed"] * mixed


def _build_row(
    *,
    domain_label: str,
    market: Market,
    platform: Platform,
    stance: str,
    index: int,
    market_index: int,
    created_at: datetime,
    collected_at: datetime,
    augmentation_version: str,
    rng: random.Random,
) -> SocialPostRecord:
    text = _message_text(domain_label, stance, rng)
    post_id = _native_post_id(
        platform=platform,
        market_id=market.market_id,
        stance=stance,
        index=index,
        market_index=market_index,
    )
    source = _source(domain_label, platform, rng)
    url = _url(platform, source, post_id)
    engagement = _engagement(platform, stance, rng)
    query = _query_for(domain_label, market)
    marker = augmentation_marker(augmentation_version)
    return SocialPostRecord(
        post_id=post_id,
        platform=platform,
        source_id=post_id,
        url=url,
        author_id=_author(platform, source, post_id),
        author_name=_author_name(platform, source, rng),
        created_at=created_at,
        text=text,
        title=text if platform is Platform.REDDIT else None,
        subreddit_or_channel=source,
        query=query,
        query_type="ui_stress_market_anchor",
        market_id_candidate=market.market_id,
        retrieval_run_id=f"{augmentation_version}-{market_index}",
        retrieval_work_id=f"{market.market_id}:{index}",
        retrieval_window_start=_window_start(market),
        retrieval_window_end=_as_utc(market.close_time),
        retrieval_lookback_days=None,
        retrieval_window_label=stance,
        retrieval_mode="ui_stress_public_social",
        temporal_valid_candidate=True,
        created_at_parse_status="parsed",
        market_id_hint=market.market_id,
        market_question_hint=market.question,
        market_window_start_hint=_window_start(market),
        market_close_time_hint=_as_utc(market.close_time),
        query_provenance=f"{marker};source=main_ui_stress_test;stance_hint={stance}",
        temporal_relation_to_market="before_close",
        leakage_eligible=True,
        exclusion_reason=None,
        engagement_score=float(engagement["score"]),
        upvotes=engagement.get("upvotes"),
        comments_count=engagement.get("comments"),
        shares_count=engagement.get("shares"),
        views_count=engagement.get("views"),
        reactions_count=engagement.get("reactions"),
        language="en",
        raw_path=None,
        collected_at=collected_at,
        is_duplicate=False,
        text_length=len(text),
    )


def _merge_generated(
    existing: list[SocialPostRecord],
    generated: list[SocialPostRecord],
    *,
    augmentation_version: str,
) -> list[SocialPostRecord]:
    marker = augmentation_marker(augmentation_version)
    kept = [row for row in existing if marker not in (row.query_provenance or "")]
    merged: list[SocialPostRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in [*kept, *generated]:
        key = (row.platform.value, row.post_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row.model_copy(update={"is_duplicate": False}))
    return merged


def _read_existing_social_records(paths: DataPaths) -> list[SocialPostRecord]:
    records = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    csv_path = paths.social_posts_parquet.with_suffix(".csv")
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        if "post_id" in frame.columns:
            frame = frame[frame["post_id"].notna()].copy()
        if "platform" in frame.columns:
            frame = frame[frame["platform"].notna()].copy()
        if "text" in frame.columns:
            frame["text"] = frame["text"].fillna("")
        if "source_id" in frame.columns and "post_id" in frame.columns:
            frame["source_id"] = frame["source_id"].fillna(frame["post_id"])
        if "text_length" in frame.columns and "text" in frame.columns:
            frame["text_length"] = frame["text_length"].fillna(frame["text"].str.len())
        csv_records = dataframe_to_models(frame, SocialPostRecord)
        by_key: dict[tuple[str, str], SocialPostRecord] = {
            (row.platform.value, row.post_id): row for row in records
        }
        for row in csv_records:
            by_key.setdefault((row.platform.value, row.post_id), row)
        records = list(by_key.values())
    return records


def _message_text(domain_label: str, stance: str, rng: random.Random) -> str:
    anchors = _ANCHORS[domain_label]
    if stance == "supports_yes":
        template = rng.choice(_YES_TEMPLATES)
    elif stance == "supports_no":
        template = rng.choice(_NO_TEMPLATES)
    elif stance == "neutral":
        template = rng.choice(_NEUTRAL_TEMPLATES)
    else:
        template = rng.choice(_MIXED_TEMPLATES)
    suffixes = (
        "Volume is picking up and the replies are split.",
        "This is exactly the kind of market where timing matters.",
        "Watching the next update closely.",
        "The deadline makes this more sensitive than a normal headline trade.",
        "I would not ignore the latest public signals here.",
    )
    return f"{template.format(anchor0=anchors[0], anchor1=anchors[1], anchor2=anchors[2], anchor3=anchors[3])} {rng.choice(suffixes)}"


def _sample_created_at(market: Market, rng: random.Random) -> datetime:
    close_time = _as_utc(market.close_time)
    start = _window_start(market)
    latest = close_time - timedelta(hours=1)
    if latest <= start:
        start = close_time - timedelta(days=30)
        latest = close_time - timedelta(hours=1)
    span_seconds = max(1, int((latest - start).total_seconds()))
    return start + timedelta(seconds=rng.randint(0, span_seconds))


def _window_start(market: Market) -> datetime:
    close_time = _as_utc(market.close_time)
    created_at = _as_utc(market.created_at) if market.created_at is not None else None
    if created_at is not None and created_at < close_time:
        return created_at
    return close_time - timedelta(days=30)


def _native_post_id(
    *,
    platform: Platform,
    market_id: str,
    stance: str,
    index: int,
    market_index: int,
) -> str:
    digest = hashlib.sha1(f"{market_id}|{stance}|{index}|{market_index}".encode()).hexdigest()
    if platform is Platform.REDDIT:
        return f"t3_{_base36(int(digest[:12], 16)).lower()[:9]}"
    channel_id = 100000 + int(digest[:6], 16) % 800000
    message_id = 1000 + market_index * 10000 + index
    return f"telegram:{channel_id}:{message_id}"


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = ""
    while value:
        value, rem = divmod(value, 36)
        out = alphabet[rem] + out
    return out


def _source(domain_label: str, platform: Platform, rng: random.Random) -> str:
    if platform is Platform.REDDIT:
        return rng.choice(_REDDIT_SOURCES[domain_label])
    return rng.choice(_TELEGRAM_SOURCES[domain_label])


def _url(platform: Platform, source: str, post_id: str) -> str:
    if platform is Platform.REDDIT:
        slug = post_id.removeprefix("t3_")
        return f"https://old.reddit.com/r/{source}/comments/{slug}/ui_stress_thread/"
    _, _, message_id = post_id.split(":")
    return f"https://t.me/{source}/{message_id}"


def _author(platform: Platform, source: str, post_id: str) -> str:
    if platform is Platform.TELEGRAM:
        return source
    return f"u_{hashlib.sha1(post_id.encode('utf-8')).hexdigest()[:8]}"


def _author_name(platform: Platform, source: str, rng: random.Random) -> str:
    if platform is Platform.TELEGRAM:
        return source
    stems = ("marketwatcher", "data_trader", "newsflow", "probabilitydesk", "eventreader")
    return f"{rng.choice(stems)}_{rng.randint(10, 9999)}"


def _engagement(platform: Platform, stance: str, rng: random.Random) -> dict[str, int]:
    base = 20 if stance in {"supports_yes", "supports_no"} else 8
    if platform is Platform.REDDIT:
        upvotes = rng.randint(base, base + 240)
        comments = rng.randint(0, 85)
        return {"score": upvotes + comments, "upvotes": upvotes, "comments": comments}
    views = rng.randint(base * 80, base * 900)
    reactions = rng.randint(0, 220)
    shares = rng.randint(0, 35)
    return {"score": reactions + shares, "views": views, "reactions": reactions, "shares": shares}


def _query_for(domain_label: str, market: Market) -> str:
    anchors = _ANCHORS[domain_label]
    return f"{anchors[0]} {anchors[1]} {market.question}"


def _reddit_share(domain_label: str) -> float:
    if domain_label in {"sports", "entertainment/culture"}:
        return 0.65
    if domain_label == "crypto/finance":
        return 0.55
    return 0.50


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        raise ValueError("datetime value is required")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def plan_to_frame(plan: AugmentationPlan) -> pd.DataFrame:
    return models_to_dataframe(plan.generated_rows)


__all__ = [
    "DEFAULT_AUGMENTATION_VERSION",
    "DEFAULT_MARKETS_PER_RUN",
    "DEFAULT_MESSAGES_PER_MARKET",
    "SELECTED_MARKETS",
    "AugmentationPlan",
    "MarketAnchor",
    "augment_main_social_dataset",
    "augmentation_marker",
    "generate_augmented_rows",
    "load_selected_markets",
    "plan_augmentation",
    "plan_to_frame",
]
