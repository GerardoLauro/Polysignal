"""Deterministic Parquet-to-Neo4j planning and loading.

Graph Loader v1 keeps planning separate from execution. ``build_load_plan``
reads canonical Parquet outputs and produces idempotent Cypher operations
without connecting to Neo4j. ``execute_load_plan`` sends those operations in
batches when a live database is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from polysignal.domain import LinkedPost, Market, Outcome, SocialPostRecord, SocialPostSegment
from polysignal.domain.enums import RelevanceLabel, StanceLabel
from polysignal.evaluation import assess_post_admissibility
from polysignal.graph import queries
from polysignal.stance.models import StanceRecord, read_stance_records
from polysignal.storage.parquet_store import read_dataframe, read_parquet
from polysignal.storage.paths import DataPaths, get_data_paths


class GraphWriteClient(Protocol):
    """Minimal client surface required by the executor."""

    def run_write(self, statement: str, rows: list[dict[str, Any]]) -> None: ...


ACCEPTED_WEAK_LINK_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class DatasetStatus:
    """Filesystem status for one planner input or intentionally skipped dataset."""

    name: str
    path: Path
    required: bool
    exists: bool
    row_count: int = 0


@dataclass(frozen=True)
class LoadOperation:
    """One ordered idempotent Cypher write plus its parameter rows."""

    name: str
    kind: str
    statement: str
    rows: tuple[dict[str, Any], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class GraphLoadPlan:
    """Offline-inspectable graph construction plan."""

    datasets: dict[str, DatasetStatus]
    operations: list[LoadOperation]
    planned_node_counts: dict[str, int]
    planned_relationship_counts: dict[str, int]
    missing_optional_datasets: list[str] = field(default_factory=list)
    skipped_components: list[str] = field(default_factory=list)
    skipped_rows: dict[str, int] = field(default_factory=dict)

    def operation(self, name: str) -> LoadOperation:
        for operation in self.operations:
            if operation.name == name:
                return operation
        raise KeyError(name)


@dataclass(frozen=True)
class GraphLoadResult:
    """Execution summary returned after a live load."""

    batches_executed: int
    rows_processed: int
    operation_rows: dict[str, int]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    return value


def _record(**values: Any) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in values.items()}


def _status(name: str, path: Path, *, required: bool, row_count: int = 0) -> DatasetStatus:
    return DatasetStatus(
        name=name,
        path=path,
        required=required,
        exists=path.exists(),
        row_count=row_count,
    )


def _parquet_row_count(path: Path) -> int:
    return len(read_dataframe(path)) if path.exists() else 0


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"required graph input {label!r} not found at {path}. "
            "Run the corresponding ingestion/linking pipeline first."
        )


def _operation(name: str, kind: str, statement: str, rows: Iterable[dict[str, Any]]) -> LoadOperation:
    return LoadOperation(name=name, kind=kind, statement=statement, rows=tuple(rows))


def _unique_rows(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row[column] for column in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _market_rows(markets: Sequence[Market]) -> list[dict[str, Any]]:
    return [
        _record(
            market_id=market.market_id,
            slug=market.slug,
            question=market.question,
            description=market.description,
            domain=market.domain,
            created_at=market.created_at,
            close_time=market.close_time,
            resolved_at=market.resolved_at,
            is_resolved=market.is_resolved,
            is_closed=market.is_closed,
            resolution=market.resolution,
            tags=market.tags,
            source_url=market.source_url,
            event_id=market.event_id,
            condition_id=market.condition_id,
            volume=market.volume,
            liquidity=market.liquidity,
        )
        for market in markets
    ]


def _outcome_rows(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not outcome.market_id:
            raise ValueError("graph outcome rows require market_id")
        rows.append(
            _record(
                market_id=outcome.market_id,
                outcome_id=outcome.outcome_id,
                label=outcome.label,
                index=outcome.index,
                current_price=outcome.current_price,
            )
        )
    return rows


def _post_rows(posts: Sequence[SocialPostRecord]) -> list[dict[str, Any]]:
    return _unique_rows(
        (
            _record(
                platform=post.platform,
                post_id=post.post_id,
                source_id=post.source_id,
                url=post.url,
                author_id=post.author_id,
                author_name=post.author_name,
                created_at=post.created_at,
                text=post.text,
                title=post.title,
                channel=post.subreddit_or_channel,
                engagement_score=post.engagement_score,
                language=post.language,
                raw_path=post.raw_path,
                collected_at=post.collected_at,
            )
            for post in posts
        ),
        ("platform", "post_id"),
    )


def _segment_index(segments: Sequence[SocialPostSegment]) -> dict[str, SocialPostSegment]:
    return {segment.segment_id: segment for segment in segments}


def _resolve_parent_post_id(
    source_post_id: str,
    *,
    parent_post_id: str | None,
    segment_id: str | None,
    segments: dict[str, SocialPostSegment],
) -> tuple[str, str | None]:
    segment = segments.get(source_post_id) or (segments.get(segment_id) if segment_id else None)
    if segment is not None:
        return segment.parent_post_id, segment.segment_id
    return parent_post_id or source_post_id, segment_id


def _link_rows(
    links: Sequence[LinkedPost],
    *,
    segments: dict[str, SocialPostSegment],
    posts: dict[tuple[str, str], SocialPostRecord],
    markets: dict[str, Market],
    skipped_rows: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for link in links:
        platform = link.platform.value
        parent_post_id, segment_id = _resolve_parent_post_id(
            link.post_id,
            parent_post_id=link.parent_post_id,
            segment_id=link.segment_id,
            segments=segments,
        )
        post = posts.get((platform, parent_post_id))
        market = markets.get(link.market_id)
        if post is None or market is None:
            skipped_rows["skipped_missing_endpoint"] = (
                skipped_rows.get("skipped_missing_endpoint", 0) + 1
            )
            continue
        label = link.final_relevance_label or link.relevance_label
        confidence = (
            link.final_confidence if link.final_confidence is not None else link.confidence
        )
        if label is RelevanceLabel.UNRELATED:
            skipped_rows["skipped_unrelated"] = skipped_rows.get("skipped_unrelated", 0) + 1
            continue
        if label is RelevanceLabel.AMBIGUOUS:
            skipped_rows["skipped_ambiguous"] = skipped_rows.get("skipped_ambiguous", 0) + 1
            continue
        if (
            label is RelevanceLabel.WEAKLY_RELEVANT
            and confidence < ACCEPTED_WEAK_LINK_MIN_CONFIDENCE
        ):
            skipped_rows["skipped_low_confidence"] = (
                skipped_rows.get("skipped_low_confidence", 0) + 1
            )
            continue
        post_created_at = link.post_created_at or post.created_at
        admissibility = assess_post_admissibility(market, post_created_at)
        if not admissibility.admissible:
            key = f"skipped_{admissibility.reason}"
            skipped_rows[key] = skipped_rows.get(key, 0) + 1
            continue
        rows.append(
            _record(
                platform=platform,
                parent_post_id=parent_post_id,
                market_id=link.market_id,
                link_id=link.link_id,
                segment_id=segment_id,
                source_linked_post_id=link.post_id,
                retrieval_query=link.query,
                relevance_label=link.relevance_label,
                final_relevance_label=label,
                final_relevance_score=(
                    link.final_relevance_score
                    if link.final_relevance_score is not None
                    else link.relevance_score
                ),
                final_confidence=confidence,
                final_method=link.final_method or link.method,
                constraint_match=link.final_constraint_match or link.constraint_match,
                created_at=link.created_at,
                source_post_created_at=post_created_at,
                accepted=True,
                temporal_valid=admissibility.temporal_valid,
                temporal_anchor=admissibility.anchor,
                temporal_anchor_kind=admissibility.anchor_kind,
            )
        )
    unique = _unique_rows(rows, ("link_id",))
    duplicates = len(rows) - len(unique)
    if duplicates:
        skipped_rows["skipped_duplicate"] = skipped_rows.get("skipped_duplicate", 0) + duplicates
    return unique


def _outcomes_by_market(outcomes: Sequence[Outcome]) -> dict[str, list[Outcome]]:
    grouped: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        if outcome.market_id:
            grouped.setdefault(outcome.market_id, []).append(outcome)
    return {
        market_id: sorted(market_outcomes, key=lambda item: item.index)
        for market_id, market_outcomes in grouped.items()
    }


def _resolve_platform(
    parent_post_id: str,
    *,
    preferred: str | None,
    post_keys: set[tuple[str, str]],
) -> str | None:
    if preferred and (preferred, parent_post_id) in post_keys:
        return preferred
    matches = [platform for platform, post_id in post_keys if post_id == parent_post_id]
    return matches[0] if len(matches) == 1 else None


def _resolve_stance_outcome(
    stance: StanceRecord,
    *,
    outcomes: dict[str, list[Outcome]],
) -> tuple[Outcome, str] | None:
    market_outcomes = outcomes.get(stance.market_id, [])
    if stance.outcome_id:
        target = next(
            (outcome for outcome in market_outcomes if outcome.outcome_id == stance.outcome_id),
            None,
        )
        if target is None:
            return None
        is_no_outcome = target.label.strip().lower() in {"no", "false"}
        relationship = (
            "SUPPORTS"
            if stance.stance_label is StanceLabel.SUPPORTS_YES or is_no_outcome
            else "OPPOSES"
        )
        return target, relationship
    if len(market_outcomes) != 2:
        return None
    wanted = "yes" if stance.stance_label is StanceLabel.SUPPORTS_YES else "no"
    target = next(
        (outcome for outcome in market_outcomes if outcome.label.strip().lower() == wanted),
        None,
    )
    return (target, "SUPPORTS") if target is not None else None


def _stance_rows(
    stances: Sequence[StanceRecord],
    *,
    segments: dict[str, SocialPostSegment],
    post_keys: set[tuple[str, str]],
    post_created_at_by_key: dict[tuple[str, str], datetime | None],
    markets: dict[str, Market],
    outcomes: dict[str, list[Outcome]],
    accepted_link_keys: set[tuple[str, str]],
    skipped_rows: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supports: list[dict[str, Any]] = []
    opposes: list[dict[str, Any]] = []
    directional = {StanceLabel.SUPPORTS_YES, StanceLabel.SUPPORTS_NO}
    for stance in stances:
        if stance.stance_label not in directional:
            skipped_rows["stance_non_directional"] = skipped_rows.get("stance_non_directional", 0) + 1
            continue
        if (stance.market_id, stance.post_id) not in accepted_link_keys:
            skipped_rows["stance_unaccepted_link"] = (
                skipped_rows.get("stance_unaccepted_link", 0) + 1
            )
            continue
        parent_post_id, segment_id = _resolve_parent_post_id(
            stance.post_id,
            parent_post_id=stance.parent_post_id,
            segment_id=stance.segment_id,
            segments=segments,
        )
        preferred_platform = stance.source_platform.value if stance.source_platform else None
        platform = _resolve_platform(parent_post_id, preferred=preferred_platform, post_keys=post_keys)
        resolved = _resolve_stance_outcome(stance, outcomes=outcomes)
        market = markets.get(stance.market_id)
        if platform is None or resolved is None or market is None:
            skipped_rows["stance_unresolved_endpoint"] = (
                skipped_rows.get("stance_unresolved_endpoint", 0) + 1
            )
            continue
        post_created_at = stance.post_created_at or post_created_at_by_key.get(
            (platform, parent_post_id)
        )
        admissibility = assess_post_admissibility(market, post_created_at)
        if not admissibility.admissible:
            key = f"stance_{admissibility.reason}"
            skipped_rows[key] = skipped_rows.get(key, 0) + 1
            continue
        outcome, relationship = resolved
        row = _record(
            platform=platform,
            parent_post_id=parent_post_id,
            market_id=stance.market_id,
            outcome_id=outcome.outcome_id,
            stance_id=stance.stance_id,
            stance_label=stance.stance_label,
            confidence=stance.confidence,
            method=stance.method,
            rationale=stance.explanation,
            created_at=stance.created_at,
            source_post_created_at=post_created_at,
            temporal_valid=admissibility.temporal_valid,
            temporal_anchor=admissibility.anchor,
            temporal_anchor_kind=admissibility.anchor_kind,
            segment_id=segment_id,
            source_stance_post_id=stance.post_id,
        )
        (supports if relationship == "SUPPORTS" else opposes).append(row)
    return _unique_rows(supports, ("stance_id",)), _unique_rows(opposes, ("stance_id",))


def build_load_plan(
    *,
    paths: DataPaths | None = None,
    segments_path: Path | None = None,
    stance_path: Path | None = None,
) -> GraphLoadPlan:
    """Read canonical Parquet datasets and return an offline graph-load plan."""
    paths = paths or get_data_paths()
    required_paths = {
        "markets": paths.polymarket_markets_parquet,
        "outcomes": paths.polymarket_outcomes_parquet,
        "social_posts": paths.social_posts_parquet,
        "linked_posts": paths.linked_posts_parquet,
    }
    for name, path in required_paths.items():
        _require(path, name)

    segment_source = segments_path or paths.social_post_segments_parquet
    stance_source = stance_path or paths.stance_labels_parquet
    if segments_path is not None:
        _require(segment_source, "social_post_segments")
    if stance_path is not None:
        _require(stance_source, "stance_labels")

    markets = read_parquet(required_paths["markets"], Market)
    outcomes = read_parquet(required_paths["outcomes"], Outcome)
    posts = read_parquet(required_paths["social_posts"], SocialPostRecord)
    links = read_parquet(required_paths["linked_posts"], LinkedPost)
    segments = (
        read_parquet(segment_source, SocialPostSegment) if segment_source.exists() else []
    )
    stances = read_stance_records(stance_source) if stance_source.exists() else []

    datasets = {
        "markets": _status("markets", required_paths["markets"], required=True, row_count=len(markets)),
        "outcomes": _status(
            "outcomes", required_paths["outcomes"], required=True, row_count=len(outcomes)
        ),
        "social_posts": _status(
            "social_posts", required_paths["social_posts"], required=True, row_count=len(posts)
        ),
        "linked_posts": _status(
            "linked_posts", required_paths["linked_posts"], required=True, row_count=len(links)
        ),
        "social_post_segments": _status(
            "social_post_segments", segment_source, required=False, row_count=len(segments)
        ),
        "stance_labels": _status(
            "stance_labels", stance_source, required=False, row_count=len(stances)
        ),
        "price_points": _status(
            "price_points",
            paths.polymarket_price_points_parquet,
            required=False,
            row_count=_parquet_row_count(paths.polymarket_price_points_parquet),
        ),
        "entities": _status(
            "entities",
            paths.entities_parquet,
            required=False,
            row_count=_parquet_row_count(paths.entities_parquet),
        ),
        "social_pressure": _status(
            "social_pressure",
            paths.social_pressure_parquet,
            required=False,
            row_count=_parquet_row_count(paths.social_pressure_parquet),
        ),
    }
    missing_optional = [
        name for name, status in datasets.items() if not status.required and not status.exists
    ]

    market_rows = _unique_rows(_market_rows(markets), ("market_id",))
    market_ids = {row["market_id"] for row in market_rows}
    market_by_id = {market.market_id: market for market in markets}
    outcome_rows = _unique_rows(_outcome_rows(outcomes), ("market_id", "outcome_id"))
    orphan_outcomes = [row for row in outcome_rows if row["market_id"] not in market_ids]
    if orphan_outcomes:
        raise ValueError(
            f"graph outcomes contain {len(orphan_outcomes)} row(s) whose market_id "
            "does not exist in markets.parquet"
        )
    post_rows = _post_rows(posts)
    post_keys = {(row["platform"], row["post_id"]) for row in post_rows}
    post_by_key = {(post.platform.value, post.post_id): post for post in posts}
    post_created_at_by_key = {
        (post.platform.value, post.post_id): post.created_at for post in posts
    }
    domain_rows = _unique_rows(({"name": row["domain"]} for row in market_rows), ("name",))
    platform_rows = _unique_rows(({"name": row["platform"]} for row in post_rows), ("name",))
    author_rows = _unique_rows(
        (
            _record(platform=row["platform"], author_id=row["author_id"], handle=row["author_name"])
            for row in post_rows
            if row["author_id"]
        ),
        ("platform", "author_id"),
    )
    contract_domain_rows = [
        _record(market_id=row["market_id"], domain=row["domain"]) for row in market_rows
    ]
    contract_outcome_rows = [
        _record(market_id=row["market_id"], outcome_id=row["outcome_id"]) for row in outcome_rows
    ]
    post_platform_rows = [
        _record(platform=row["platform"], post_id=row["post_id"]) for row in post_rows
    ]
    author_post_rows = [
        _record(platform=row["platform"], author_id=row["author_id"], post_id=row["post_id"])
        for row in post_rows
        if row["author_id"]
    ]

    skipped_rows: dict[str, int] = {}
    segment_lookup = _segment_index(segments)
    linked_to_rows = _link_rows(
        links,
        segments=segment_lookup,
        posts=post_by_key,
        markets=market_by_id,
        skipped_rows=skipped_rows,
    )
    accepted_link_keys = {
        (row["market_id"], row["source_linked_post_id"]) for row in linked_to_rows
    }
    supports_rows, opposes_rows = _stance_rows(
        stances,
        segments=segment_lookup,
        post_keys=post_keys,
        post_created_at_by_key=post_created_at_by_key,
        markets=market_by_id,
        outcomes=_outcomes_by_market(outcomes),
        accepted_link_keys=accepted_link_keys,
        skipped_rows=skipped_rows,
    )

    operations = [
        _operation("upsert_contracts", "node", queries.UPSERT_CONTRACTS, market_rows),
        _operation("upsert_domains", "node", queries.UPSERT_DOMAINS, domain_rows),
        _operation("upsert_outcomes", "node", queries.UPSERT_OUTCOMES, outcome_rows),
        _operation("upsert_platforms", "node", queries.UPSERT_PLATFORMS, platform_rows),
        _operation("upsert_posts", "node", queries.UPSERT_POSTS, post_rows),
        _operation("upsert_authors", "node", queries.UPSERT_AUTHORS, author_rows),
        _operation(
            "link_contract_domains", "relationship", queries.LINK_CONTRACT_DOMAINS, contract_domain_rows
        ),
        _operation(
            "link_contract_outcomes",
            "relationship",
            queries.LINK_CONTRACT_OUTCOMES,
            contract_outcome_rows,
        ),
        _operation(
            "link_post_platforms", "relationship", queries.LINK_POST_PLATFORMS, post_platform_rows
        ),
        _operation("link_author_posts", "relationship", queries.LINK_AUTHOR_POSTS, author_post_rows),
        _operation("link_post_contracts", "relationship", queries.LINK_POST_CONTRACTS, linked_to_rows),
        _operation(
            "link_post_supports_outcome",
            "relationship",
            queries.LINK_POST_SUPPORTS_OUTCOME,
            supports_rows,
        ),
        _operation(
            "link_post_opposes_outcome",
            "relationship",
            queries.LINK_POST_OPPOSES_OUTCOME,
            opposes_rows,
        ),
    ]
    skipped_components = [
        "PricePoint nodes deferred: price history remains analytical Parquet.",
        "Entity nodes and MENTIONS edges deferred: no production NER dataset.",
        "Topic and Narrative nodes deferred: no extraction contract.",
        "Social Pressure Index integration deferred.",
    ]
    if not stance_source.exists():
        skipped_components.append("Directional stance edges skipped: stance_labels.parquet absent.")

    return GraphLoadPlan(
        datasets=datasets,
        operations=operations,
        planned_node_counts={
            "Contract": len(market_rows),
            "Domain": len(domain_rows),
            "Outcome": len(outcome_rows),
            "Platform": len(platform_rows),
            "Post": len(post_rows),
            "Author": len(author_rows),
        },
        planned_relationship_counts={
            "BELONGS_TO": len(contract_domain_rows),
            "HAS_OUTCOME": len(contract_outcome_rows),
            "POSTED_ON": len(post_platform_rows),
            "WROTE": len(author_post_rows),
            "LINKED_TO": len(linked_to_rows),
            "SUPPORTS": len(supports_rows),
            "OPPOSES": len(opposes_rows),
        },
        missing_optional_datasets=missing_optional,
        skipped_components=skipped_components,
        skipped_rows=skipped_rows,
    )


def _batches(rows: tuple[dict[str, Any], ...], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def execute_load_plan(
    client: GraphWriteClient,
    plan: GraphLoadPlan,
    *,
    batch_size: int = 500,
) -> GraphLoadResult:
    """Execute a planned graph load in deterministic operation order."""
    batches_executed = 0
    rows_processed = 0
    operation_rows: dict[str, int] = {}
    for operation in plan.operations:
        operation_rows[operation.name] = operation.row_count
        for batch in _batches(operation.rows, batch_size):
            client.run_write(operation.statement, batch)
            batches_executed += 1
            rows_processed += len(batch)
    return GraphLoadResult(
        batches_executed=batches_executed,
        rows_processed=rows_processed,
        operation_rows=operation_rows,
    )


__all__ = [
    "ACCEPTED_WEAK_LINK_MIN_CONFIDENCE",
    "DatasetStatus",
    "GraphLoadPlan",
    "GraphLoadResult",
    "LoadOperation",
    "build_load_plan",
    "execute_load_plan",
]
