# Telegram public-channel ingestion

Telegram is the immediate social source while Reddit credentials are pending.
PolySignal uses Telethon in user-session mode and fetches only explicit public
channel/group usernames. It does not enumerate dialogs or collect private chats.

## Local setup

Install the optional dependency:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[telegram]"
```

Store credentials only in the ignored root `.env` file:

```text
POLYSIGNAL_TELEGRAM_API_ID=<telegram-api-id>
POLYSIGNAL_TELEGRAM_API_HASH=<telegram-api-hash>
POLYSIGNAL_TELEGRAM_SESSION_NAME=polysignal_telegram
POLYSIGNAL_TELEGRAM_SESSION_DIR=data/cache/telegram
```

Never print, document, or commit the API hash. Telethon session files are stored
under the ignored `data/cache/telegram/` directory.

Create a local channel config from the committed template:

```powershell
Copy-Item config\telegram_channels.example.yml config\telegram_channels.yml
notepad config\telegram_channels.yml
```

Only add public channel/group usernames you explicitly want to collect.

## Authenticate and ingest

The first authentication can prompt interactively for phone number, login code,
and two-factor password:

```powershell
.venv\Scripts\python.exe -m polysignal social telegram auth
```

Fetch one channel into `data/raw/social/telegram/`, then normalize it:

```powershell
.venv\Scripts\python.exe -m polysignal social telegram fetch-channel --channel examplecrypto --limit 200
.venv\Scripts\python.exe -m polysignal social telegram normalize
```

Fetch and normalize all enabled config entries:

```powershell
.venv\Scripts\python.exe -m polysignal social telegram ingest --config config\telegram_channels.yml --limit-per-channel 200
```

Optional filters:

```powershell
.venv\Scripts\python.exe -m polysignal social telegram fetch-channel --channel examplecrypto --query bitcoin --limit 200
.venv\Scripts\python.exe -m polysignal social telegram ingest --config config\telegram_channels.yml --start-date 2026-05-01T00:00:00Z --end-date 2026-05-31T23:59:59Z
```

Dry-run does not require credentials or network access:

```powershell
.venv\Scripts\python.exe -m polysignal social telegram ingest --config config\telegram_channels.example.yml --dry-run
```

## Linking

Telegram normalization appends deduplicated rows to
`data/normalized/social/social_posts.parquet`. Direct channel posts do not need
`social_queries.parquet`: Branch 05 retains query provenance when available and
otherwise uses bounded candidate prefiltering. Explicit `full-scan` mode remains
available for debugging and recall checks.

```powershell
.venv\Scripts\python.exe -m polysignal linking score-candidates
```

Output lands at `data/normalized/social/linked_posts.parquet`.
`social inspect-posts` safely renders empty/media-only messages as `<empty>`.
For manual review, export sorted links to CSV:

```powershell
.venv\Scripts\python.exe -m polysignal linking inspect-links --limit 100 --export-csv data\normalized\social\linked_posts_sorted.csv
```

Telegram ingestion is working locally. Branch 05 remains relevance-only; it
does not perform stance detection.

The bounded 05F public-channel check fetched 300 posts, generated 222 optimized
candidate rows in `auto` mode, and compared them with a separate 15,000-row
full scan. The review export is
`data/analytical/linking/manual_review_candidates.csv`.

### Multi-topic digest segmentation (Branch 05G)

Long Telegram digest messages that bundle many independent news items
are split into smaller segments before scoring so anchors from different
sections cannot combine into a false positive. The original
`social_posts.parquet` is preserved untouched; segments live in
`data/normalized/social/social_post_segments.parquet`.

```powershell
.venv\Scripts\python.exe -m polysignal social segment-posts
.venv\Scripts\python.exe -m polysignal linking score-candidates
.venv\Scripts\python.exe -m polysignal linking export-manual-review
```

`linking score-candidates` prefers segments when the segments parquet
exists (toggle with `--use-segments` / `--no-segments`). Each row in
`linked_posts.parquet` carries optional `segment_id`, `parent_post_id`,
`segment_index`, and `segment_text_preview` fields so the dashboard and
manual reviewers can trace a verdict back to the exact paragraph used
for scoring. See `docs/BRANCH_05_LINKING.md` §10A for the full schema
and the phrase-anchor gate.

## Limitations

- Public channels/groups or explicit configured targets only.
- No private-chat collection and no limit bypassing.
- Telegram rate limits still apply; keep limits modest.
- Channel data is noisy and requires Branch 05 semantic linking.
- Telegram ingestion is not stance detection.
