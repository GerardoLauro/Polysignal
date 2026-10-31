"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def offline_network_guard(monkeypatch):
    """Default tests may use loopback, never remote collectors or paid services."""
    import os
    import socket

    if any(os.environ.get(name) == "1" for name in
           ("POLYSIGNAL_RUN_NETWORK_TESTS", "POLYSIGNAL_RUN_LIVE_REDDIT_TESTS")):
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def permitted(address):
        return isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}

    def connect(sock, address):
        if not permitted(address):
            raise AssertionError("External network disabled in offline tests")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not permitted(address):
            raise AssertionError("External network disabled in offline tests")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)

from polysignal.domain import (
    Engagement,
    Market,
    MarketDomain,
    Outcome,
    Platform,
    SocialPost,
)


@pytest.fixture
def binary_market() -> Market:
    return Market(
        market_id="m1",
        slug="will-candidate-x-win-2025",
        question="Will candidate X win the 2025 election?",
        domain=MarketDomain.POLITICS,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        outcomes=[
            Outcome(outcome_id="yes", label="Yes", index=0),
            Outcome(outcome_id="no", label="No", index=1),
        ],
    )


@pytest.fixture
def post_about_market() -> SocialPost:
    return SocialPost(
        post_id="p1",
        platform=Platform.REDDIT,
        author_id="u1",
        author_handle="alice",
        created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        text="Candidate X is collapsing in the polls; this looks like a loss.",
        engagement=Engagement(likes=10, replies=2),
    )


@pytest.fixture
def post_unrelated() -> SocialPost:
    return SocialPost(
        post_id="p2",
        platform=Platform.REDDIT,
        author_id="u2",
        author_handle="bob",
        created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        text="Pizza recipes are great in winter.",
    )
