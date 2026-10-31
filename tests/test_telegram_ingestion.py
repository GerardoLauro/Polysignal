"""Offline tests for Telegram public-channel ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from polysignal.cli.main import app
from polysignal.config import Settings
from polysignal.dataset import TelegramSearchPlanEntry
from polysignal.domain import MarketQuery, Platform, SocialPostRecord
from polysignal.ingestion.social import telegram_normalizer
from polysignal.ingestion.social.base import SearchQuery
from polysignal.ingestion.social.normalizer import build_record
from polysignal.ingestion.social.raw_store import SocialRawWriter
from polysignal.ingestion.social.telegram import (
    TelegramIngestor,
    raw_message_to_social_post,
    telegram_message_to_raw_dict,
)
from polysignal.ingestion.social.telegram_normalizer import (
    load_telegram_channels,
    normalize_telegram_raw_directory,
    search_telegram_market_plan,
)
from polysignal.linking import CandidateMode, fixture_markets, run_linking_from_records
from polysignal.storage.parquet_store import read_parquet, write_parquet
from polysignal.storage.paths import get_data_paths


@dataclass
class _Reaction:
    count: int


@dataclass
class _FakeChannel:
    id: int = 12345
    username: str = "examplecrypto"
    title: str = "Example Crypto"
    broadcast: bool = True
    megagroup: bool = False


@dataclass
class _FakeMessage:
    id: int = 42
    message: str = "Bitcoin could hit $150k by December 2026."
    date: datetime = datetime(2026, 5, 31, 12, 0)
    views: int | None = 1200
    forwards: int | None = 4
    replies: Any = None
    reactions: Any = None
    media: Any = None
    post_author: str | None = None
    grouped_id: int | None = None
    edit_date: datetime | None = None


def _message(**kwargs: Any) -> _FakeMessage:
    values = {
        "replies": SimpleNamespace(replies=3),
        "reactions": SimpleNamespace(results=[_Reaction(count=5), _Reaction(count=2)]),
    }
    values.update(kwargs)
    return _FakeMessage(**values)


def _settings(api_id: str = "123", api_hash: str = "hash") -> Settings:
    return Settings(
        telegram={
            "api_id": api_id,
            "api_hash": api_hash,
            "session_name": "test-session",
        }
    )


def _raw(message: _FakeMessage | None = None) -> dict[str, Any]:
    return telegram_message_to_raw_dict(
        message or _message(),
        _FakeChannel(),
        requested_channel="examplecrypto",
    )


def test_telegram_raw_fixture_normalizes_with_stable_id_and_utc() -> None:
    post = raw_message_to_social_post(_raw())
    assert post is not None
    assert post.platform is Platform.TELEGRAM
    assert post.post_id == "telegram:12345:42"
    assert post.url == "https://t.me/examplecrypto/42"
    assert post.created_at.tzinfo is timezone.utc
    assert post.engagement.views == 1200
    assert post.engagement.reposts == 4
    assert post.engagement.replies == 3
    assert post.engagement.reactions == 7


def test_telegram_empty_and_media_only_messages_are_safe() -> None:
    empty = _raw(_message(message="", views=None, forwards=None, replies=None, reactions=None))
    media = _raw(
        _message(
            id=43,
            message="",
            views=None,
            forwards=None,
            replies=None,
            reactions=None,
            media=SimpleNamespace(),
        )
    )
    assert raw_message_to_social_post(empty) is None
    normalized_media = raw_message_to_social_post(media)
    assert normalized_media is not None
    assert normalized_media.text == ""


def test_social_inspect_posts_handles_media_only_telegram_row(tmp_path: Path) -> None:
    media = raw_message_to_social_post(
        _raw(
            _message(
                id=43,
                message="",
                views=None,
                forwards=None,
                replies=None,
                reactions=None,
                media=SimpleNamespace(),
            )
        )
    )
    assert media is not None
    target = tmp_path / "social_posts.parquet"
    write_parquet(
        [
            build_record(
                post=media,
                query=None,
                market_id_candidate=None,
                raw_path=None,
                collected_at=datetime(2026, 5, 31, tzinfo=timezone.utc),
            )
        ],
        target,
    )
    result = CliRunner().invoke(
        app,
        ["social", "inspect-posts", "--posts-path", str(target), "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "<empty>" in result.output
    assert ":43" in result.output


def test_telegram_raw_directory_appends_compatible_deduplicated_parquet(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    raw_dir = paths.raw_social("telegram") / "run"
    writer = SocialRawWriter(directory=raw_dir, platform="telegram")
    writer.write_post("bitcoin", "telegram-channel-examplecrypto", "telegram:12345:42", _raw())
    writer.write_post("btc", "telegram-channel-examplecrypto", "telegram:12345:42", _raw())
    writer.write_manifest()

    records = normalize_telegram_raw_directory(raw_dir, paths=paths)
    assert len(records) == 2
    record = records[0]
    assert record.platform is Platform.TELEGRAM
    assert record.post_id == "telegram:12345:42"
    assert record.market_id_candidate is None
    assert record.query_type == "telegram_channel_search"
    assert record.subreddit_or_channel == "examplecrypto"
    assert record.views_count == 1200
    assert record.reactions_count == 7
    assert record.created_at.tzinfo is not None

    normalize_telegram_raw_directory(raw_dir, paths=paths)
    loaded = read_parquet(paths.social_posts_parquet, SocialPostRecord)
    assert len(loaded) == 2
    assert {row.query for row in loaded} == {"bitcoin", "btc"}


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.started = False
        self.disconnected = False
        self.kwargs: dict[str, Any] = {}

    def start(self) -> None:
        self.started = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_entity(self, channel: str) -> _FakeChannel:
        assert channel == "examplecrypto"
        return _FakeChannel()

    def iter_messages(self, entity: _FakeChannel, **kwargs: Any):
        self.kwargs = kwargs
        return iter([_message()])


def test_telegram_fetch_uses_injected_client_without_network() -> None:
    fake = _FakeTelegramClient()
    ingestor = TelegramIngestor(
        settings=_settings(),
        channel="examplecrypto",
        telegram_client=fake,
    )
    until = datetime(2026, 6, 1, tzinfo=timezone.utc)
    posts, raws = ingestor.search_raw(SearchQuery(text="bitcoin", limit=10, until=until))
    assert len(posts) == len(raws) == 1
    assert fake.started is True
    assert fake.kwargs == {
        "limit": 10,
        "search": "bitcoin",
        "offset_date": until + timedelta(microseconds=1),
    }


def test_telegram_missing_credentials_error_is_clear() -> None:
    ingestor = TelegramIngestor(settings=_settings(api_id="", api_hash=""), channel="examplecrypto")
    try:
        list(ingestor.search(SearchQuery(text="bitcoin")))
    except RuntimeError as exc:
        assert "POLYSIGNAL_TELEGRAM_API_ID" in str(exc)
    else:
        raise AssertionError("live fetch must reject missing Telegram credentials")


def test_telegram_config_loads_only_enabled_channels(tmp_path: Path) -> None:
    config = tmp_path / "channels.yml"
    config.write_text(
        "channels:\n"
        "  - username: examplecrypto\n"
        "    enabled: true\n"
        "  - username: examplepolitics\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    assert [channel.username for channel in load_telegram_channels(config)] == ["examplecrypto"]


def test_telegram_config_parses_rich_fields(tmp_path: Path) -> None:
    config = tmp_path / "channels.yml"
    config.write_text(
        "channels:\n"
        "  - username: '@cointelegraph'\n"
        "    name: Cointelegraph\n"
        "    domain: crypto\n"
        "    language: en\n"
        "    limit: 250\n"
        "    notes: news feed\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    [channel] = load_telegram_channels(config)
    assert channel.username == "cointelegraph"  # '@' stripped
    assert channel.domain == "crypto"
    assert channel.domain_tags == ("crypto",)
    assert channel.language == "en"
    assert channel.limit == 250
    assert channel.notes == "news feed"


def test_fetch_telegram_channels_skips_failing_channel_and_continues(tmp_path: Path) -> None:
    from polysignal.domain import SocialPost
    from polysignal.ingestion.social.telegram import TelegramAPIError

    class _FakeIngestor:
        def __init__(self) -> None:
            self.closed = False

        def fetch_channel_raw(self, channel: str, query: SearchQuery):
            if channel == "private_channel":
                raise TelegramAPIError("not a channel or public-style group")
            post = SocialPost(
                post_id=f"telegram:{channel}:1",
                platform=Platform.TELEGRAM,
                author_id=channel,
                author_handle=channel,
                created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
                text=f"message from {channel}",
            )
            raw = {
                "channel": {"username": channel, "id": channel, "title": channel},
                "message": {"id": 1, "text": f"message from {channel}",
                            "date": "2026-05-01T00:00:00+00:00"},
            }
            return [post], [raw]

        def close(self) -> None:
            self.closed = True

    fake = _FakeIngestor()
    result = telegram_normalizer.fetch_telegram_channels(
        [
            telegram_normalizer.TelegramChannel(username="good_channel"),
            telegram_normalizer.TelegramChannel(username="private_channel"),
            telegram_normalizer.TelegramChannel(username="another_good"),
        ],
        ingestor=fake,
        limit=5,
        paths=get_data_paths(root=tmp_path),
    )
    assert result.posts_collected == 2
    assert {name for name, _ in result.succeeded} == {"good_channel", "another_good"}
    skipped_names = {name for name, _ in result.skipped}
    assert "private_channel" in skipped_names
    assert fake.closed is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["extra"]["channels_attempted"] == [
        "good_channel",
        "private_channel",
        "another_good",
    ]
    assert manifest["extra"]["channels_succeeded"][0]["messages_collected"] == 1
    assert manifest["extra"]["channels_skipped"] == [
        {
            "handle": "private_channel",
            "reason": "TelegramAPIError: not a channel or public-style group",
        }
    ]


def test_telegram_cli_help_and_dry_run_need_no_credentials(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["social", "telegram", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "fetch-channel" in help_result.output
    result = runner.invoke(
        app,
        [
            "social",
            "telegram",
            "fetch-channel",
            "--channel",
            "examplecrypto",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert not paths_exist(tmp_path)


def test_telegram_cli_live_fetch_reports_missing_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        telegram_normalizer,
        "TelegramIngestor",
        lambda: TelegramIngestor(settings=_settings(api_id="", api_hash="")),
    )
    result = CliRunner().invoke(
        app,
        [
            "social",
            "telegram",
            "fetch-channel",
            "--channel",
            "examplecrypto",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "POLYSIGNAL_TELEGRAM_API_ID" in result.output


def paths_exist(root: Path) -> bool:
    return get_data_paths(root=root).raw_social("telegram").exists()


def test_linking_consumes_direct_telegram_record_without_query_provenance(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    raw_dir = paths.raw_social("telegram") / "run"
    writer = SocialRawWriter(directory=raw_dir, platform="telegram")
    writer.write_post("", "telegram-channel-examplecrypto", "telegram:12345:42", _raw())
    writer.write_manifest()
    records = normalize_telegram_raw_directory(raw_dir, paths=paths)
    markets = fixture_markets()
    rows = run_linking_from_records(records, markets)
    assert 0 < len(rows) < len(markets)
    assert all(row.platform is Platform.TELEGRAM for row in rows)
    full_scan_rows = run_linking_from_records(
        records, markets, candidate_mode=CandidateMode.FULL_SCAN
    )
    assert len(full_scan_rows) == len(markets)


def test_telegram_normalization_reuses_social_query_provenance(tmp_path: Path) -> None:
    paths = get_data_paths(root=tmp_path)
    paths.ensure()
    write_parquet(
        [
            MarketQuery(
                market_id="m-btc-150k",
                query="bitcoin",
                query_type="entity",
                rank=0,
                generated_at=datetime(2026, 5, 31, tzinfo=timezone.utc),
            )
        ],
        paths.social_queries_parquet,
    )
    raw_dir = paths.raw_social("telegram") / "run"
    writer = SocialRawWriter(directory=raw_dir, platform="telegram")
    writer.write_post("bitcoin", "telegram-channel-examplecrypto", "telegram:12345:42", _raw())
    writer.write_manifest()
    records = normalize_telegram_raw_directory(raw_dir, paths=paths)
    assert records[0].market_id_candidate == "m-btc-150k"
    assert records[0].query_type == "entity"
    assert len(run_linking_from_records(records, fixture_markets())) == 1


def test_targeted_market_search_persists_provenance_and_resumes(tmp_path: Path) -> None:
    class _TargetedFakeIngestor:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def fetch_channel_raw(self, channel: str, query: SearchQuery):
            self.calls += 1
            assert channel == "examplecrypto"
            assert query.market_id == "m-btc"
            assert query.since == datetime(2026, 5, 1, tzinfo=timezone.utc)
            assert query.until == datetime(2026, 5, 8, tzinfo=timezone.utc)
            post = raw_message_to_social_post(_raw())
            assert post is not None
            return [post], [_raw()]

        def close(self) -> None:
            self.closed = True

    paths = get_data_paths(root=tmp_path)
    entry = TelegramSearchPlanEntry(
        market_id="m-btc",
        work_id="work-btc",
        channel_id="examplecrypto",
        query="bitcoin",
        query_type="entity",
        priority=0,
        temporal_anchor=datetime(2026, 5, 8, tzinfo=timezone.utc),
        temporal_anchor_kind="close_time",
        lookback_days=7,
        window_label="0-7d",
        search_window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        search_window_end=datetime(2026, 5, 8, tzinfo=timezone.utc),
    )
    channel = telegram_normalizer.TelegramChannel(username="examplecrypto", domain="crypto")
    fake = _TargetedFakeIngestor()
    first = search_telegram_market_plan([entry], [channel], ingestor=fake, paths=paths)
    assert first.posts_collected == 1
    assert first.work_items_completed == 1
    records = normalize_telegram_raw_directory(first.raw_dir, paths=paths)
    assert len(records) == 1
    assert records[0].market_id_candidate == "m-btc"
    assert records[0].retrieval_work_id == "work-btc"
    assert records[0].retrieval_window_label == "0-7d"
    assert records[0].temporal_valid_candidate is True

    resumed = _TargetedFakeIngestor()
    second = search_telegram_market_plan([entry], [channel], ingestor=resumed, paths=paths)
    assert second.work_items_skipped == 1
    assert resumed.calls == 0
