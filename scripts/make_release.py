"""Build a source-only Polysignal archive for GitHub or handoff.

Includes source, tests, report, documentation, templates and small CSV examples.
Complete data is excluded unless --include-data is explicitly requested for a
private export. Credentials, sessions, caches and raw corpora stay excluded.
The credential-pattern scan is a precaution, not a guarantee of secret absence.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = "Polysignal"

# ---------------------------------------------------------------------------
# What goes in.
# ---------------------------------------------------------------------------
TOP_LEVEL_FILES = [
    "README.md", "LICENSE", "pyproject.toml", "requirements.txt", "Makefile",
    "docker-compose.yml", ".gitignore", ".gitattributes", ".env.example", "run_app.cmd", "run_app.sh",
    "LEGGIMI.md", "MANIFEST.in", "THIRD_PARTY_NOTICES.md", "requirements-runtime.lock",
    "requirements-dev.lock", "Traccia_2 (1).pdf",
]
# Recursively included source/report/doc trees (minus the prune rules below).
TOP_LEVEL_DIRS = [".github", "src", "scripts", "webapp", "tests", "docs", "config", "configs", "pdf_latex"]

# Curated data subset: only the artifacts the dashboard reads + the exported
# dataset deliverable. Everything else under data/ is excluded by omission.
DATA_INCLUDE = [
    "data/dataset_export",                                  # the dataset deliverable
    "data/analytical/timelines",
    "data/analytical/features",
    "data/analytical/analytics",
    "data/analytical/backtests",
    "data/analytical/processing",
    "data/analytical/linking",
    "data/analytical/curated_analysis_markets.parquet",
    "data/analytical/curated_analysis_summary.json",
    "data/analytical/evidence/accepted_evidence.parquet",  # not the giant rejected set
    "data/analytical/evidence/evidence_summary.json",
    "data/normalized/polymarket",
    "data/normalized/social/stance_labels.parquet",
    "data/normalized/social/post_enrichment.parquet",
    "data/normalized/social/post_comments.parquet",
    "data/normalized/social/topics.parquet",
    "data/reports/ps_data.json",
    "data/reports/usefulness_report.json",
    "data/graph_exports",
]

# ---------------------------------------------------------------------------
# What never ships.
# ---------------------------------------------------------------------------
PRUNE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".ipynb_checkpoints", "node_modules", ".git",
}
# Specific files to drop even inside an included tree.
EXCLUDE_FILES = {
    "config/telegram_channels.yml",   # real channel list (keep only the .example)
}
EXCLUDE_SUFFIXES = {
    ".pyc", ".pyo", ".session", ".session-journal", ".log",
    ".aux", ".bbl", ".blg", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk",
}
EXCLUDE_NAMES = {".env", ".DS_Store", "Thumbs.db"}

# Heavy export tables are available only with explicit --include-data.
# --lean also drops these files in that private-export mode.
HEAVY_DATASET_FILES = {
    "data/dataset_export/social_posts.parquet",
    "data/dataset_export/links.parquet",
}

# --- extra trimming applied ONLY to the lean (send-this-one) archive ----------
# The lean archive is the "run the dashboard" package; the report is sent
# separately, and the many development-convenience scripts are not part of the
# deliverable. So under --lean we drop the whole LaTeX project and keep only the
# handful of scripts a recipient actually needs.
LEAN_EXCLUDE_DIRS = {"pdf_latex"}
LEAN_SCRIPTS_KEEP = {"serve_app.py", "build_ps_data.py", "verify_dataset.py"}

# Token shapes that must never appear in a shipped text file. We match only
# genuine high-entropy key shapes -- the known provider prefixes, or "sk-"
# followed by 32+ *contiguous* alphanumerics (a real secret) -- so that
# hyphenated market slugs like "sk-win-the-uefa-champions-league" do not trip it.
TOKEN_RE = re.compile(
    r"sk-(?:proj|ant|live)-[A-Za-z0-9_-]{20,}"  # provider-prefixed keys
    r"|sk-[A-Za-z0-9]{32,}"                       # contiguous high-entropy secret
    r"|AUTH_TOKEN\s*=\s*[A-Za-z0-9]{24,}"         # an actual token value assignment
)
SCAN_SUFFIXES = {
    ".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini",
    ".js", ".jsx", ".mjs", ".html", ".css", ".tex", ".bib", ".sh", ".cmd",
    ".json", ".example", "",
}


def _should_prune_dir(path: Path) -> bool:
    return any(part in PRUNE_DIR_NAMES or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)


def _excluded_file(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDE_FILES:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def _iter_files(lean: bool = False, include_data: bool = False) -> list[Path]:
    skip = HEAVY_DATASET_FILES if lean else set()
    files: list[Path] = []
    for name in TOP_LEVEL_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
    for d in TOP_LEVEL_DIRS:
        if lean and d in LEAN_EXCLUDE_DIRS:
            continue
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not (p.is_file() and not _should_prune_dir(p.parent) and not _excluded_file(p)):
                continue
            if lean and d == "scripts" and p.name not in LEAN_SCRIPTS_KEEP:
                continue
            files.append(p)
    for entry in (DATA_INCLUDE if include_data else ["data/README.md", "data/examples"]):
        p = ROOT / entry
        if p.is_file():
            if not _excluded_file(p):
                files.append(p)
        elif p.is_dir():
            for q in p.rglob("*"):
                if q.is_file() and not _should_prune_dir(q.parent) and not _excluded_file(q):
                    files.append(q)
    if skip:
        files = [p for p in files if p.relative_to(ROOT).as_posix() not in skip]
    # Stable, de-duplicated order.
    return sorted(set(files), key=lambda x: x.relative_to(ROOT).as_posix())


def _scan_for_secrets(files: list[Path]) -> list[str]:
    hits: list[str] = []
    for p in files:
        if p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in TOKEN_RE.finditer(text):
            hits.append(f"{p.relative_to(ROOT).as_posix()} :: possible token (value suppressed)")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="list files; do not write the zip")
    ap.add_argument("--lean", action="store_true",
                    help="send-this-one archive: drop the two multi-GB exported tables, the "
                         "whole pdf_latex/ report, and all dev-convenience scripts "
                         "(keep only serve_app.py, build_ps_data.py, verify_dataset.py)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-data", action="store_true", help="Explicit private bundle with original pipeline artifacts; never needed for GitHub")
    args = ap.parse_args()

    default_out = "release/polysignal_lean.zip" if args.lean else "release/polysignal_submission.zip"
    out_rel = args.out or default_out

    files = _iter_files(lean=args.lean, include_data=args.include_data)
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"PolySignal release: {len(files)} files, {total_bytes / 1e6:.1f} MB uncompressed")

    secrets = _scan_for_secrets(files)
    if secrets:
        print("\nABORTING -- secret-shaped tokens found in files slated for release:", file=sys.stderr)
        for h in secrets:
            print(f"  ! {h}", file=sys.stderr)
        return 2
    print("Secret scan: clean (no API-key-shaped tokens in any shipped text file).")

    if args.dry_run:
        for p in files:
            print(f"  {p.relative_to(ROOT).as_posix()}  ({p.stat().st_size:,} B)")
        return 0

    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = [f"{p.relative_to(ROOT).as_posix()}\t{p.stat().st_size}" for p in files]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, arcname=f"{ARCHIVE_PREFIX}/{p.relative_to(ROOT).as_posix()}")
        zf.writestr(f"{ARCHIVE_PREFIX}/RELEASE_MANIFEST.txt", "\n".join(manifest_lines))

    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out}  ({size_mb:.1f} MB, {len(files)} files)")
    print("Excluded by design: .env, *.session, .venv/, .tools/, _cleanup_trash/, caches,")
    print("                    data/raw, data/staging, data/cache, data/*_backup*, the")
    print("                    multi-GB intermediate corpora. Complete dashboard data must be restored separately.")
    if args.lean:
        print("Lean extras dropped: pdf_latex/ (whole report -- send main.pdf separately) and")
        print("                     all dev-convenience scripts (kept: serve_app.py,")
        print("                     build_ps_data.py, verify_dataset.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
