"""Manual resumable Reddit + X comment/reply collection.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\collect_social_comments.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from polysignal.ingestion.social.comment_backfill import (
    CommentBackfillOptions,
    run_comment_backfill,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect sampled Reddit comments and authenticated X replies."
    )
    parser.add_argument(
        "--platform",
        choices=("all", "reddit", "twitter"),
        default="all",
    )
    parser.add_argument("--reddit-max-posts", type=int, default=100)
    parser.add_argument("--twitter-max-posts", type=int, default=50)
    parser.add_argument("--comments-per-post", type=int, default=10)
    parser.add_argument("--rate-limit-seconds", type=float, default=3.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/social_enrichment_sources.example.yaml"),
    )
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not fetch; rebuild comment_sample_count and the combined table.",
    )
    parser.add_argument(
        "--force-post-update",
        action="store_true",
        help="Update Reddit posts even if the Scrapling collector lock is active.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(
            "data/staging/social_comments/reports/social_comments_collect.log"
        ),
    )
    return parser


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def main() -> int:
    args = _parser().parse_args()
    _configure_logging(args.log_path)
    if args.comments_per_post < 1:
        raise SystemExit("--comments-per-post must be >= 1")
    if args.reddit_max_posts < 0 or args.twitter_max_posts < 0:
        raise SystemExit("post limits must be >= 0 (0 means unlimited)")

    options = CommentBackfillOptions(
        platform=args.platform,
        config_path=args.config,
        reddit_max_posts=args.reddit_max_posts,
        twitter_max_posts=args.twitter_max_posts,
        comments_per_post=args.comments_per_post,
        rate_limit_seconds=args.rate_limit_seconds,
        reset_progress=args.reset_progress,
        metadata_only=args.metadata_only,
        force_post_update=args.force_post_update,
        dry_run=args.dry_run,
    )
    logging.info(
        "START social comment backfill platform=%s reddit_posts=%s "
        "twitter_posts=%s comments_per_post=%s dry_run=%s",
        options.platform,
        options.reddit_max_posts,
        options.twitter_max_posts,
        options.comments_per_post,
        options.dry_run,
    )
    result = run_comment_backfill(options)
    for item in result.platforms:
        logging.info(
            "%s selected=%s attempted=%s success=%s failed=%s "
            "sampled=%s added=%s total=%s metadata_updated=%s",
            item.platform.upper(),
            item.selected_posts,
            item.attempted_posts,
            item.successful_posts,
            item.failed_posts,
            item.sampled_comments,
            item.rows_added,
            item.total_comments,
            item.post_metadata_updated,
        )
    logging.info(
        "DONE combined_comments=%s summary=%s",
        result.combined_comments,
        options.summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
