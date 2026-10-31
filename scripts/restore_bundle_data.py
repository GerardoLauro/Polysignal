"""Restore private data archives using the bundled manifest; datasets stay outside Git."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_destination(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or relative.parts[0] != "data"
            or ".." in relative.parts or "\\" in name or ":" in name
            or relative.as_posix() != name):
        raise ValueError(f"Unsafe archive path: {name}")
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes destination")
    return target


def restore(archives, destination, manifest, dashboard_only=False, include_all=False, verify_only=False):
    destination = Path(destination).resolve()
    selected = [item for item in manifest["archives"] if
                item["name"].startswith("01-") or
                (not dashboard_only and item["name"].startswith("02-")) or
                (include_all and item["name"].startswith(("03-", "05-")))]
    pending = []
    # Verify all selected sources and existing destinations before any writes.
    for item in selected:
        name = item["name"]
        if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
            raise ValueError("Unsafe archive filename in manifest")
        archive = Path(archives) / name
        if not archive.is_file():
            raise FileNotFoundError(f"Private archive not supplied: {archive}. Datasets are not included in the GitHub source ZIP.")
        if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
            raise ValueError(f"Archive hash/size mismatch: {name}")
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            expected = {entry["path"]: entry for entry in item["files"]}
            if len(members) != len(expected) or {entry.filename for entry in members} != set(expected):
                raise ValueError(f"Unexpected or duplicate archive members: {name}")
            if len({entry.filename.casefold() for entry in members}) != len(members):
                raise ValueError("Case-insensitive member collision")
            for entry in members:
                target = safe_destination(destination, entry.filename)
                record = expected[entry.filename]
                if entry.is_dir() or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Directory or symbolic-link member not allowed")
                if entry.file_size != record["bytes"]:
                    raise ValueError(f"Member size mismatch: {entry.filename}")
                with source.open(entry) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                        raise ValueError(f"Member hash mismatch: {entry.filename}")
                if target.exists():
                    if target.is_file() and sha256(target) == record["sha256"]:
                        continue
                    raise FileExistsError(f"Existing different file preserved: {target}. Use a clean destination.")
                pending.append((archive, entry.filename, target, record["sha256"]))
    if verify_only:
        return len(pending)
    for archive, name, target, digest in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".bundle-part")
        created = False
        try:
            with zipfile.ZipFile(archive) as source, source.open(name) as incoming, temporary.open("xb") as outgoing:
                created = True
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if sha256(temporary) != digest:
                raise ValueError(f"Member changed during restore: {name}")
            temporary.replace(target)
        finally:
            if created and temporary.exists():
                temporary.unlink()
    return len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT.parent / "Dati-e-materiali")
    parser.add_argument("--destination", type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dashboard-only", action="store_true")
    group.add_argument("--all", action="store_true", help="Include private intermediates and raw collection data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "docs/BUNDLE_DATA.json").read_text(encoding="utf-8"))
    try:
        count = restore(args.archives, args.destination, manifest, args.dashboard_only, args.all, args.verify_only)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{error}\n")
    print(f"{'Verified' if args.verify_only else 'Restored'} {count} files. Private data remains ignored by Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
