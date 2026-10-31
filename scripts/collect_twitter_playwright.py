"""Authenticated Twitter/X collector using Playwright + an existing Chromium.

Why this exists
---------------
As of 2026, X (Twitter) serves profile *timelines* only to authenticated
sessions. Every no-auth path is blocked from a normal network:
  * ``x.com/<user>`` plain HTTP returns a JS shell with **zero** tweets;
  * ``syndication.twitter.com`` returns HTTP 429 (IP-level rate limit);
  * ``cdn.syndication.twimg.com`` is token-gated and returns an empty body;
  * a logged-out headless browser loads the profile header but renders **no**
    timeline tweets (the GraphQL ``UserTweets`` call needs auth).

So this collector drives a real Chromium (already installed via Playwright)
using **your** X session cookies. Cookies are read from the environment / the
local ``.env`` file and are NEVER written to any versioned file.

How to provide credentials (local, untracked ``.env`` only)
-----------------------------------------------------------
Log into x.com in a browser, open DevTools -> Application -> Cookies -> x.com,
copy the two values and add them to ``.env``::

    X_AUTH_TOKEN=<value of the `auth_token` cookie>
    X_CT0=<value of the `ct0` cookie>

Then run::

    .venv\\Scripts\\python.exe scripts\\collect_twitter_playwright.py [target_records]

Data is written to ``data/staging/twitter_web/social_posts.parquet`` (staging
only - never canonical ``data/normalized``), the same path the monitor reads.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from polysignal.ingestion.social.enrichment import (
    csv_sibling,
    persist_records,
    social_posts_to_records,
)
from polysignal.ingestion.social.enrichment_config import TwitterSource
from polysignal.ingestion.social.twitter_scrapling_adapter import twitter_raw_to_social_post

# Domain-focused account set (sports / crypto / finance) matching the 300-market
# study. X profile timelines only expose recent tweets and X rate-limits after a
# handful of accounts per session, so the pool is deliberately broad: each run
# scrapes a subset successfully and repeated runs (hourly loop / 30-min monitor)
# cover the rest over time as new tweets appear.
# Ordered sports -> finance -> crypto so the rate-limit tail (X throttles after
# ~15 accounts/session) lands on the already-collected crypto handles (harmless
# dupes) rather than starving an under-collected domain. Rotate the leading block
# between passes for full coverage over a long session.
TWITTER_ACCOUNTS: list[tuple[str, str, list[str]]] = [
    # sports (collect first — these were the rate-limited tail last pass)
    ("espn", "sports", ["sports", "nba", "nfl", "soccer"]),
    ("FabrizioRomano", "sports", ["sports", "football", "soccer", "transfers"]),
    ("BleacherReport", "sports", ["sports", "nba", "nfl", "mlb"]),
    ("NBA", "sports", ["sports", "nba", "basketball"]),
    ("NFL", "sports", ["sports", "nfl", "football"]),
    ("brfootball", "sports", ["sports", "soccer", "football"]),
    ("MLB", "sports", ["sports", "mlb", "baseball"]),
    ("F1", "sports", ["sports", "formula1", "motorsport"]),
    ("ChampionsLeague", "sports", ["sports", "soccer", "football"]),
    ("SkySports", "sports", ["sports", "football", "formula1"]),
    # finance / macro
    ("Bloomberg", "finance", ["finance", "macro", "markets", "news"]),
    ("Reuters", "finance", ["news", "finance", "geopolitics"]),
    ("WSJ", "finance", ["finance", "macro", "markets", "news"]),
    ("CNBC", "finance", ["finance", "markets", "stocks", "news"]),
    ("FinancialTimes", "finance", ["finance", "macro", "markets"]),
    ("federalreserve", "finance", ["macro", "rates", "policy"]),
    ("AP", "news", ["news", "politics", "world"]),
    ("TheEconomist", "finance", ["finance", "macro", "politics"]),
    ("YahooFinance", "finance", ["finance", "markets", "stocks"]),
    ("MarketWatch", "finance", ["finance", "markets", "stocks"]),
    ("BBCWorld", "news", ["news", "politics", "geopolitics"]),
    ("CNN", "news", ["news", "politics", "us"]),
    ("nytimes", "news", ["news", "politics", "us"]),
    ("politico", "news", ["politics", "elections", "us"]),
    ("ForeignAffairs", "news", ["geopolitics", "politics", "world"]),
    # crypto (already well-collected — fine as the dupe-heavy tail)
    ("Polymarket", "crypto", ["crypto", "markets", "prediction"]),
    ("VitalikButerin", "crypto", ["crypto", "ethereum", "web3"]),
    ("coinbase", "crypto", ["crypto", "markets", "exchange"]),
    ("binance", "crypto", ["crypto", "markets", "exchange"]),
    ("CoinDesk", "crypto", ["crypto", "news", "markets"]),
    ("Cointelegraph", "crypto", ["crypto", "news", "markets"]),
    ("BitcoinMagazine", "crypto", ["crypto", "bitcoin"]),
    ("saylor", "crypto", ["crypto", "bitcoin", "markets"]),
    ("Variety", "entertainment", ["entertainment", "film", "tv", "awards"]),
    ("DEADLINE", "entertainment", ["entertainment", "film", "tv"]),
    ("DiscussingFilm", "entertainment", ["entertainment", "film", "boxoffice"]),
    ("TechCrunch", "tech", ["technology", "startups", "ai"]),
    ("verge", "tech", ["technology", "consumer", "ai"]),
    ("OpenAI", "tech", ["technology", "ai", "research"]),
    ("GoogleDeepMind", "tech", ["technology", "ai", "research"]),
    # --- 2026-06-21: widened the pool to grow POST volume (politics/geopolitics
    # --- carry most market evidence; bad/renamed handles just skip gracefully).
    ("WhiteHouse", "news", ["politics", "us", "policy"]),
    ("POTUS", "news", ["politics", "us"]),
    ("sentdefender", "news", ["geopolitics", "osint", "world"]),
    ("Faytuks", "news", ["geopolitics", "breaking_news", "world"]),
    ("WarMonitors", "news", ["geopolitics", "breaking_news", "world"]),
    ("AFP", "news", ["news", "world", "geopolitics"]),
    ("BBCBreaking", "news", ["news", "breaking_news", "world"]),
    ("disclosetv", "news", ["news", "politics", "breaking_news"]),
    ("spectatorindex", "news", ["news", "geopolitics", "world"]),
    ("zerohedge", "finance", ["finance", "macro", "markets"]),
    ("KobeissiLetter", "finance", ["finance", "macro", "markets"]),
    ("FirstSquawk", "finance", ["finance", "macro", "breaking_news"]),
    ("unusual_whales", "finance", ["finance", "stocks", "markets"]),
    ("DeItaone", "finance", ["finance", "macro", "breaking_news"]),
    ("APompliano", "crypto", ["crypto", "bitcoin", "markets"]),
    ("DocumentingBTC", "crypto", ["crypto", "bitcoin"]),
    ("wojespn", "sports", ["sports", "nba", "basketball"]),
    ("goal", "sports", ["sports", "soccer", "football"]),
    ("TheAthletic", "sports", ["sports", "nba", "nfl", "soccer"]),
    ("OptaJoe", "sports", ["sports", "soccer", "football"]),
]

STAGING = Path("data/staging/twitter_web/social_posts.parquet")
COLLECTION_METHOD = "twitter_authenticated_web"

# DOM-extraction JS: pull every rendered tweet article in one round-trip.
_EXTRACT_JS = r"""
() => {
  const out = [];
  for (const art of document.querySelectorAll('article[data-testid="tweet"]')) {
    const textEl = art.querySelector('div[data-testid="tweetText"]');
    const text = textEl ? textEl.innerText : "";
    if (!text) continue;
    const timeEl = art.querySelector('time');
    const created_at = timeEl ? timeEl.getAttribute('datetime') : null;
    let tweet_id = "", url = "";
    for (const a of art.querySelectorAll('a[href*="/status/"]')) {
      const m = a.getAttribute('href').match(/\/([A-Za-z0-9_]+)\/status\/(\d+)/);
      if (m) { url = "https://x.com" + a.getAttribute('href').split('?')[0]; tweet_id = m[2]; break; }
    }
    out.push({ text, created_at, tweet_id, url });
  }
  return out;
}
"""


def _load_env_file(path: Path) -> None:
    """Best-effort load of KEY=VALUE pairs from a local .env into os.environ."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_cookies() -> list[dict[str, object]]:
    _load_env_file(Path(".env"))
    auth_token = os.environ.get("X_AUTH_TOKEN") or os.environ.get("TWITTER_AUTH_TOKEN")
    ct0 = os.environ.get("X_CT0") or os.environ.get("TWITTER_CT0")
    if not auth_token or not ct0:
        return []
    cookies: list[dict[str, object]] = []
    for domain in (".x.com", ".twitter.com"):
        cookies.append({"name": "auth_token", "value": auth_token, "domain": domain,
                        "path": "/", "httpOnly": True, "secure": True})
        cookies.append({"name": "ct0", "value": ct0, "domain": domain,
                        "path": "/", "secure": True})
    return cookies


def _rotated_accounts() -> list[tuple[str, str, list[str]]]:
    """Rotate the leading accounts so X rate-limit tails vary by pass."""
    if not TWITTER_ACCOUNTS:
        return []
    try:
        offset = int(os.environ.get("POLYSIGNAL_X_ACCOUNT_OFFSET", "0"))
    except ValueError:
        offset = 0
    offset %= len(TWITTER_ACCOUNTS)
    accounts = TWITTER_ACCOUNTS[offset:] + TWITTER_ACCOUNTS[:offset]
    try:
        max_accounts = int(os.environ.get("POLYSIGNAL_X_MAX_ACCOUNTS", "0"))
    except ValueError:
        max_accounts = 0
    return accounts[:max_accounts] if max_accounts > 0 else accounts


def _collect_account(page, username: str, max_items: int) -> list[dict]:
    seen: dict[str, dict] = {}
    page.goto(f"https://x.com/{username}", wait_until="domcontentloaded", timeout=45000)
    # Wait for the timeline to render (auth required for these to appear).
    for _ in range(8):
        if page.locator("article[data-testid='tweet']").count() > 0:
            break
        time.sleep(1.5)
    stale_scrolls = 0
    for _ in range(60):  # scroll budget per account
        for raw in page.evaluate(_EXTRACT_JS):
            key = raw.get("tweet_id") or raw.get("text", "")[:80]
            if key and key not in seen:
                raw["username"] = username
                seen[key] = raw
        if len(seen) >= max_items:
            break
        before = len(seen)
        page.mouse.wheel(0, 4000)
        time.sleep(1.8)
        stale_scrolls = stale_scrolls + 1 if len(seen) == before else 0
        if stale_scrolls >= 4:  # no new tweets after several scrolls -> done
            break
    return list(seen.values())[:max_items]


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. `pip install playwright`.", file=sys.stderr)
        return 2

    cookies = _read_cookies()
    if not cookies:
        print(
            "ERROR: X session cookies not found.\n"
            "  Add X_AUTH_TOKEN and X_CT0 to your local .env (untracked), then re-run.\n"
            "  See the module docstring for how to copy them from your browser.",
            file=sys.stderr,
        )
        return 3

    target_records = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    accounts = _rotated_accounts()
    max_items = max(50, target_records // len(accounts))
    STAGING.parent.mkdir(parents=True, exist_ok=True)
    moment = datetime.now(tz=timezone.utc)
    run_id = f"twitter_pw_{moment.strftime('%Y%m%dT%H%M%SZ')}"

    sources = {
        username: TwitterSource(
            name=f"twitter_{username.lower()}",
            username=username,
            url=f"https://x.com/{username}",
            domains=tuple(domains),
            priority="high" if category in {"crypto", "sports"} else "medium",
        )
        for username, category, domains in accounts
    }

    collected: list[tuple] = []
    print(f"Authenticated X collection: {len(sources)} accounts, "
          f"max_items_per_source={max_items} -> {STAGING}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            viewport={"width": 1280, "height": 2200}, locale="en-US",
        )
        ctx.add_cookies(cookies)
        for username, source in sources.items():
            page = ctx.new_page()
            try:
                raws = _collect_account(page, username, max_items)
                n = 0
                for raw in raws:
                    post = twitter_raw_to_social_post(
                        raw, source=source, collection_method=COLLECTION_METHOD,
                        source_url=source.url,
                    )
                    if post is not None:
                        collected.append((post, None))
                        n += 1
                print(f"  {source.name:28s} {'ok' if n else 'empty':6s} tweets={n}")
            except Exception as exc:  # one bad account must not kill the run
                print(f"  {source.name:28s} failed  {type(exc).__name__}: {str(exc)[:90]}")
            finally:
                with suppress(Exception):
                    page.close()
        browser.close()

    records = social_posts_to_records(
        collected, collection_method=COLLECTION_METHOD,
        query_type="twitter_authenticated_web", run_id=run_id, collected_at=moment,
    )
    merged, rows_added, dropped = persist_records(records, posts_path=STAGING, append=True)
    print(f"TOTAL new_records={len(records)} rows_added={rows_added} "
          f"dropped_dupes={dropped} staged_total={len(merged)}")
    print(f"CSV mirror: {csv_sibling(STAGING)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
