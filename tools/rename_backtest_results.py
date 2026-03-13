#!/usr/bin/env python3
"""Rename generic Freqtrade backtest artifacts to include the strategy name."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".tsv", ".txt", ".md", ".csv"}
GENERIC_PATTERN = "backtest-result-*.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename generic backtest-result artifacts to "
            "<strategy_name>-<timestamp> and update text references."
        )
    )
    parser.add_argument(
        "--root",
        default="user_data/backtest_results",
        help="Backtest results directory to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without renaming files.",
    )
    return parser.parse_args()


def sanitize_strategy_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    if not sanitized:
        raise ValueError(f"Could not sanitize strategy name {name!r}")
    return sanitized


def load_strategy_name(meta_path: Path) -> str:
    data = json.loads(meta_path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Unexpected metadata structure in {meta_path}")
    return sanitize_strategy_name(next(iter(data)))


def build_new_base(meta_path: Path, strategy_name: str) -> str:
    suffix = meta_path.name.removeprefix("backtest-result-").removesuffix(".meta.json")
    return f"{strategy_name}-{suffix}"


def rename_pairs(root: Path, dry_run: bool) -> tuple[dict[str, str], list[str]]:
    replacements: dict[str, str] = {}
    actions: list[str] = []

    for meta_path in sorted(root.rglob(GENERIC_PATTERN)):
        strategy_name = load_strategy_name(meta_path)
        new_base = build_new_base(meta_path, strategy_name)
        old_base = meta_path.name.removesuffix(".meta.json")

        old_zip = meta_path.with_suffix("").with_suffix(".zip")
        new_meta = meta_path.with_name(f"{new_base}.meta.json")
        new_zip = meta_path.with_name(f"{new_base}.zip")

        if new_meta.exists() and new_meta != meta_path:
            raise FileExistsError(f"Refusing to overwrite existing file: {new_meta}")
        if old_zip.exists() and new_zip.exists() and new_zip != old_zip:
            raise FileExistsError(f"Refusing to overwrite existing file: {new_zip}")

        replacements[str(meta_path.relative_to(root.parent))] = str(new_meta.relative_to(root.parent))
        replacements[str(old_base)] = new_base

        if old_zip.exists():
            replacements[str(old_zip.relative_to(root.parent))] = str(new_zip.relative_to(root.parent))

        actions.append(f"{meta_path} -> {new_meta}")
        if old_zip.exists():
            actions.append(f"{old_zip} -> {new_zip}")

        if not dry_run:
            meta_path.rename(new_meta)
            if old_zip.exists():
                old_zip.rename(new_zip)

    return replacements, actions


def update_text_references(root: Path, replacements: dict[str, str], dry_run: bool) -> list[str]:
    updated_files: list[str] = []
    if not replacements:
        return updated_files

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".last_result.json":
            continue

        original = path.read_text()
        updated = original
        for old, new in replacements.items():
            updated = updated.replace(old, new)

        if updated == original:
            continue

        updated_files.append(str(path))
        if not dry_run:
            path.write_text(updated)

    return updated_files


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    replacements, actions = rename_pairs(root, dry_run=args.dry_run)
    updated_files = update_text_references(root, replacements, dry_run=args.dry_run)

    print(f"Renamed artifacts: {len(actions)}")
    for action in actions[:20]:
        print(action)
    if len(actions) > 20:
        print(f"... {len(actions) - 20} more rename actions")

    print(f"Updated text files: {len(updated_files)}")
    for path in updated_files[:20]:
        print(path)
    if len(updated_files) > 20:
        print(f"... {len(updated_files) - 20} more updated text files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
