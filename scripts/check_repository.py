#!/usr/bin/env python3
"""Offline release checks. Never prints matched credential values."""
import ast
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote, urlsplit

SKIP = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "build", "dist", "out", ".hypothesis", "release", ".mypy_cache", ".ruff_cache"}
TEXT = {".py", ".md", ".txt", ".json", ".toml", ".sh", ".ps1", ".bat", ".yml", ".lock", ".yaml", ".cmd", ".jsx", ".js", ".html", ".tex", ".bib", ".csv", ".in"}
SECRETS = [re.compile(r"\bsk-(?:proj|ant|live)-[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9]{32,}"),
           re.compile(r"\bgh[pousr]_[A-Za-z0-9]{25,}"),
           re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
           re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]


def check(root: Path) -> dict:
    errors = []
    counts = {"files": 0, "python": 0, "json": 0, "notebooks": 0, "relative_links": 0, "latex_references": 0, "html_assets": 0}
    seen = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP or part.endswith(".egg-info") for part in rel.parts) or not p.is_file():
            continue
        counts["files"] += 1
        if p.is_symlink():
            errors.append(f"Symbolic link in release: {rel}")
        key = rel.as_posix().lower()
        if key in seen:
            errors.append(f"Case collision: {rel} / {seen[key]}")
        seen[key] = rel
        if p.name == ".env" or p.name.startswith(".env.") and p.name != ".env.example":
            errors.append(f"Local secret configuration included: {rel}")
        if any(re.search(r'[<>:"|?*]', part) or part.endswith((" ", ".")) for part in rel.parts):
            errors.append(f"Nonportable path: {rel}")
        if any(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part) for part in rel.parts):
            errors.append(f"Windows-reserved path: {rel}")
        if p.stat().st_size > 25 * 1024**2:
            errors.append(f"Public repository contains file above browser upload limit: {rel}")
        if p.suffix not in TEXT and p.name not in {"LICENSE", ".env.example", ".gitignore", ".gitattributes"}:
            if p.suffix == ".ipynb":
                try:
                    notebook = json.loads(p.read_text(encoding="utf-8"))
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                            errors.append(f"Notebook has saved outputs: {rel}")
                    counts["notebooks"] += 1
                except (ValueError, KeyError) as exc:
                    errors.append(f"Invalid notebook: {rel}: {type(exc).__name__}")
            continue
        try:
            raw = p.read_bytes()
            text = raw.decode("utf-8")
            if text.startswith("\ufeff"):
                errors.append(f"Unexpected UTF-8 BOM: {rel}")
            if p.suffix in {".sh", ".py"} and b"\r\n" in raw:
                errors.append(f"CRLF in portable source: {rel}")
            if any(pattern.search(text) for pattern in SECRETS):
                errors.append(f"Possible credential found (value suppressed): {rel}")
            if p.suffix == ".py":
                ast.parse(text, filename=str(rel), feature_version=(3, 11))
                counts["python"] += 1
            if p.suffix == ".json":
                json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                counts["json"] += 1
            if p.suffix == ".toml":
                tomllib.loads(text)
            if p.suffix == ".tex":
                base = root / "pdf_latex"
                for command, name in re.findall(r"\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}", text):
                    suffix = {"input": ".tex", "include": ".tex", "bibliography": ".bib", "includegraphics": ".pdf"}[command]
                    target = Path(name) if Path(name).suffix else Path(name + suffix)
                    candidates = [p.parent / target, base / target]
                    if command == "includegraphics":
                        candidates.append(base / "figures" / target)
                    counts["latex_references"] += 1
                    if not any(candidate.is_file() for candidate in candidates):
                        errors.append(f"Broken LaTeX reference: {rel} -> {name}")
            if p.suffix == ".html":
                for name in re.findall(r'<(?:script|link)\b[^>]*(?:src|href)=["\']([^"\']+)', text):
                    url = urlsplit(name)
                    if url.scheme or url.netloc:
                        continue
                    counts["html_assets"] += 1
                    if not (p.parent / unquote(url.path)).is_file():
                        errors.append(f"Missing frontend asset: {rel} -> {name}")
            if p.suffix == ".md":
                for match in re.finditer(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text):
                    url = urlsplit(match.group(1).strip("<>"))
                    if url.scheme or url.netloc or not url.path:
                        continue
                    target = (p.parent / unquote(url.path)).resolve()
                    counts["relative_links"] += 1
                    if not target.is_relative_to(root.resolve()) or not target.exists():
                        errors.append(f"Broken relative link: {rel} -> {url.path}")
        except (UnicodeError, ValueError, SyntaxError) as exc:
            errors.append(f"Invalid text/syntax: {rel}: {type(exc).__name__}")
    for required in ["README.md", "LICENSE", ".gitignore", ".env.example", "pyproject.toml", "requirements-runtime.lock", "THIRD_PARTY_NOTICES.md"]:
        if not (root / required).is_file():
            errors.append(f"Missing required repository file: {required}")
    return {"counts": counts, "errors": errors}


def main():
    root = Path(__file__).resolve().parents[1]
    result = check(root)
    print(json.dumps(result, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    sys.exit(main())
