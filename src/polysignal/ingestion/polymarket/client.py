"""Polymarket HTTP client.

Reads from the public Gamma (market metadata) and CLOB (price history)
APIs. The client is intentionally thin: HTTP + retry + pagination,
nothing more. Filtering and normalization live in
:mod:`polysignal.ingestion.polymarket.binary_filter` and
:mod:`polysignal.ingestion.polymarket.normalizer`.

Key design points
-----------------
* **Read-only.** No POSTs, no order placement.
* **Retry with backoff.** ``tenacity`` is already a project dependency;
  we use ``retry_if_exception_type`` for transient HTTP / network errors
  and exponential backoff bounded by a small attempt count. Permanent
  errors (4xx) are not retried.
* **Pagination.** ``iter_markets`` walks ``offset`` until the API stops
  returning rows or until ``limit`` is satisfied.
* **No raw persistence.** That is the caller's job (so the pipeline
  controls the run-scoped layout). The client exposes the parsed JSON
  pages so the caller can dump them verbatim before normalization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polysignal.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PolymarketAPIError(RuntimeError):
    """Raised when the Polymarket API returns an unrecoverable error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_TRANSIENT_EXC = (httpx.TransportError, httpx.RemoteProtocolError, httpx.TimeoutException)

# HTTP statuses we treat as "this lookup variant was rejected, try the
# next one" inside :meth:`PolymarketClient.fetch_market`. Anything else
# (5xx, 401/403, transport failures, …) propagates so we never hide a
# real API outage behind a None / "not found" response.
_SOFT_LOOKUP_STATUSES: frozenset[int] = frozenset({400, 404, 422})


def _is_retryable_status(exc: BaseException) -> bool:
    """Predicate: retry HTTP 5xx and 429 responses."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, _TRANSIENT_EXC)


class PolymarketClient:
    """Read-only HTTP client over Polymarket's Gamma + CLOB APIs.

    Parameters
    ----------
    settings:
        A pre-built :class:`Settings`. Falls back to ``get_settings()``.
    http_client:
        Optional pre-built ``httpx.Client``. Tests inject a client
        backed by ``httpx.MockTransport`` so the suite never touches
        the network.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        cfg = (settings or get_settings()).polymarket
        self._cfg = cfg
        self._own_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=cfg.request_timeout_seconds,
            headers={"User-Agent": cfg.user_agent},
        )

    # --------------------------------------------------------------- context
    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ----------------------------------------------------------------- core
    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        try:
            return self._get_with_retry(url, params=params)
        except httpx.HTTPStatusError as exc:
            raise PolymarketAPIError(
                f"GET {url} → {exc.response.status_code}: {exc.response.text[:300]}",
                status_code=exc.response.status_code,
            ) from exc
        except _TRANSIENT_EXC as exc:
            raise PolymarketAPIError(f"GET {url} failed after retries: {exc}") from exc

    @retry(
        reraise=True,
        retry=retry_if_exception_type((*_TRANSIENT_EXC, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        stop=stop_after_attempt(4),
    )
    def _get_with_retry(self, url: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.get(url, params=params)
        except _TRANSIENT_EXC as exc:
            logger.warning("polymarket: transient error on GET %s: %s", url, exc)
            raise
        if response.status_code >= 400:
            if _is_retryable_status_response(response):
                logger.warning(
                    "polymarket: retryable %s on GET %s — body=%s",
                    response.status_code,
                    url,
                    response.text[:200],
                )
                response.raise_for_status()
            raise PolymarketAPIError(
                f"GET {url} → {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise PolymarketAPIError(
                f"GET {url} returned non-JSON body (status={response.status_code})"
            ) from exc

    # ------------------------------------------------------------ endpoints
    def fetch_markets_page(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one page of markets from the Gamma API.

        Returns the parsed JSON list. ``limit`` is capped by the
        upstream API (typically 100 / 500).
        """
        url = f"{self._cfg.gamma_api_base.rstrip('/')}/markets"
        params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
        if extra_params:
            params.update(extra_params)
        try:
            payload = self._get(url, params=params)
        except RetryError as exc:  # pragma: no cover — tenacity reraise covers this
            raise PolymarketAPIError(f"GET {url} exhausted retries: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("markets") or []
        if not isinstance(payload, list):
            raise PolymarketAPIError(
                f"GET {url} returned unexpected shape: {type(payload).__name__}"
            )
        return payload

    def fetch_events_page(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one page of events from the Gamma API.

        Events are useful for discovery because each event carries its
        associated markets. Walking events avoids taking many markets
        from the first event family in the raw ``/markets`` ordering.
        """
        url = f"{self._cfg.gamma_api_base.rstrip('/')}/events"
        params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
        if extra_params:
            params.update(extra_params)
        try:
            payload = self._get(url, params=params)
        except RetryError as exc:  # pragma: no cover - tenacity reraise covers this
            raise PolymarketAPIError(f"GET {url} exhausted retries: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("events") or []
        if not isinstance(payload, list):
            raise PolymarketAPIError(
                f"GET {url} returned unexpected shape: {type(payload).__name__}"
            )
        return payload

    def iter_markets(
        self,
        *,
        limit: int | None = None,
        page_size: int = 100,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        """Yield ``(page_index, page_payload)`` tuples until exhausted.

        ``limit`` caps the **total** number of markets returned across
        pages, not the page size. Stops when the upstream returns an
        empty page or fewer than ``page_size`` rows.
        """
        if limit is not None and limit <= 0:
            return
        seen = 0
        page_index = 0
        offset = 0
        while True:
            this_page_size = page_size
            if limit is not None:
                remaining = limit - seen
                if remaining <= 0:
                    return
                this_page_size = min(page_size, remaining)
            page = self.fetch_markets_page(
                limit=this_page_size,
                offset=offset,
                extra_params=extra_params,
            )
            if not page:
                return
            yield page_index, page
            seen += len(page)
            offset += len(page)
            page_index += 1
            if len(page) < this_page_size:
                return

    def iter_events(
        self,
        *,
        limit: int | None = None,
        page_size: int = 100,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        """Yield ``(page_index, page_payload)`` tuples from ``/events``."""
        if limit is not None and limit <= 0:
            return
        seen = 0
        page_index = 0
        offset = 0
        while True:
            this_page_size = page_size
            if limit is not None:
                remaining = limit - seen
                if remaining <= 0:
                    return
                this_page_size = min(page_size, remaining)
            page = self.fetch_events_page(
                limit=this_page_size,
                offset=offset,
                extra_params=extra_params,
            )
            if not page:
                return
            yield page_index, page
            seen += len(page)
            offset += len(page)
            page_index += 1
            if len(page) < this_page_size:
                return

    def fetch_market(self, market_id: str) -> dict[str, Any] | None:
        """Fetch one market by condition id / slug / numeric id.

        The Gamma API exposes single-market lookups via the listing
        endpoint with filtering parameters. Two quirks are handled here:

        * the ``condition_ids`` / ``slug`` filters silently **exclude
          closed markets** unless ``closed=true`` is also passed, so each
          variant is retried with the closed flag when the first answer
          is empty;
        * unknown filter parameters are silently ignored (Gamma then
          returns the head of the default listing), so every returned
          payload is verified against the requested reference before it
          is accepted.

        Only "bad lookup parameter" responses (HTTP 400 / 404 / 422)
        move us to the next variant — any other error (5xx, 401/403,
        transport failure, non-JSON body, …) propagates so the caller
        cannot mistake an API outage for "no such market".

        Returns ``None`` only when Gamma successfully answered at least
        one lookup variant and no verified match was found.
        """
        url = f"{self._cfg.gamma_api_base.rstrip('/')}/markets"
        last_soft_error: PolymarketAPIError | None = None
        queried_successfully = False
        for param_name in ("condition_ids", "slug", "id"):
            # First pass without the ``closed`` filter (the active-market case).
            # The ``closed=true`` retry only fires when this returns 200 but no
            # verified match, which is exactly how Gamma hides settled markets
            # from the bare condition/slug filters.
            for extra in ({}, {"closed": "true"}):
                params: dict[str, Any] = {param_name: market_id, "limit": 2, **extra}
                try:
                    payload = self._get(url, params=params)
                except PolymarketAPIError as exc:
                    if exc.status_code in _SOFT_LOOKUP_STATUSES:
                        logger.debug(
                            "polymarket.fetch_market: lookup %s returned %s, trying next variant",
                            params,
                            exc.status_code,
                        )
                        last_soft_error = exc
                        break  # this param spelling is rejected — skip its closed retry
                    raise
                queried_successfully = True
                if isinstance(payload, dict):
                    payload = payload.get("data") or payload.get("markets") or []
                if not isinstance(payload, list):
                    break
                matched = next(
                    (
                        row
                        for row in payload
                        if isinstance(row, dict) and _market_matches_ref(row, market_id)
                    ),
                    None,
                )
                if matched is not None:
                    return matched
                # No verified match: fall through to the ``closed=true`` retry
                # (settled markets are hidden from the bare filter) and then to
                # the next param spelling. Rows that don't match are never
                # accepted, so a Gamma listing-head response can't fool us.

        if not queried_successfully and last_soft_error is not None:
            # Every lookup variant was rejected — Gamma never returned a
            # 200, so we cannot honestly report "market not found".
            raise PolymarketAPIError(
                "fetch_market: every lookup variant rejected by Gamma "
                f"(last status={last_soft_error.status_code}): {last_soft_error}",
                status_code=last_soft_error.status_code,
            ) from last_soft_error
        return None

    def fetch_price_history(
        self,
        token_id: str,
        *,
        interval: str = "max",
        fidelity: int = 60,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch the price history envelope for a single CLOB token.

        Returns the raw envelope (e.g. ``{"history": [...]}``) so the
        caller can persist it and pass it to the normalizer.

        .. warning:: ``interval=max`` only covers roughly the last three
           weeks of a market's life once it has settled. For resolved
           markets use :meth:`fetch_price_history_range`, which walks the
           full lifetime in bounded ``startTs``/``endTs`` windows.
        """
        url = f"{self._cfg.clob_api_base.rstrip('/')}/prices-history"
        params: dict[str, Any] = {
            "market": token_id,
            "interval": interval,
            "fidelity": int(fidelity),
        }
        if extra_params:
            params.update(extra_params)
        try:
            payload = self._get(url, params=params)
        except RetryError as exc:  # pragma: no cover
            raise PolymarketAPIError(f"GET {url} exhausted retries: {exc}") from exc
        if isinstance(payload, list):
            return {"history": payload}
        if not isinstance(payload, dict):
            raise PolymarketAPIError(
                f"GET {url} returned unexpected shape: {type(payload).__name__}"
            )
        return payload

    # The CLOB rejects startTs/endTs windows much longer than two weeks
    # ("interval is too long"), so the full-lifetime fetch walks 14-day
    # chunks and merges them.
    PRICE_HISTORY_CHUNK_SECONDS: int = 14 * 86400

    def fetch_price_history_range(
        self,
        token_id: str,
        *,
        start_ts: int,
        end_ts: int,
        fidelity: int = 60,
        chunk_seconds: int | None = None,
        pause_seconds: float = 0.3,
    ) -> dict[str, Any]:
        """Fetch the complete price history between two unix timestamps.

        Walks the CLOB ``/prices-history`` endpoint in bounded
        ``startTs``/``endTs`` windows (the API rejects long windows) and
        returns one merged ``{"history": [...]}`` envelope, deduplicated
        by timestamp and sorted ascending. This is the reliable path for
        **resolved** markets, where ``interval=max`` returns at most the
        final ~3 weeks (or nothing at all for older markets).
        """
        import time as _time

        if end_ts <= start_ts:
            return {"history": []}
        url = f"{self._cfg.clob_api_base.rstrip('/')}/prices-history"
        step = int(chunk_seconds or self.PRICE_HISTORY_CHUNK_SECONDS)
        merged: dict[int, dict[str, Any]] = {}
        cursor = int(start_ts)
        while cursor < end_ts:
            window_end = min(cursor + step, int(end_ts))
            params = {
                "market": token_id,
                "startTs": cursor,
                "endTs": window_end,
                "fidelity": int(fidelity),
            }
            try:
                payload = self._get(url, params=params)
            except RetryError as exc:  # pragma: no cover
                raise PolymarketAPIError(f"GET {url} exhausted retries: {exc}") from exc
            rows = payload.get("history") if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                for row in rows:
                    ts = row.get("t")
                    if isinstance(ts, (int, float)):
                        merged[int(ts)] = row
            cursor = window_end
            if pause_seconds > 0 and cursor < end_ts:
                _time.sleep(pause_seconds)
        history = [merged[ts] for ts in sorted(merged)]
        return {"history": history}


def _market_matches_ref(payload: dict[str, Any], ref: str) -> bool:
    """Whether a Gamma market payload actually answers the requested lookup."""
    wanted = ref.strip().lower()
    for key in ("conditionId", "condition_id", "slug", "id"):
        value = payload.get(key)
        if value is not None and str(value).strip().lower() == wanted:
            return True
    return False


def _is_retryable_status_response(response: httpx.Response) -> bool:
    return response.status_code >= 500 or response.status_code == 429


__all__ = ["PolymarketAPIError", "PolymarketClient"]
