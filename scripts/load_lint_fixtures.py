#!/usr/bin/env python3
"""Load lint fixtures from eval/lint_fixtures/ into wiki/ for demo or e2e testing.

Standalone CLI script. Safe to run multiple times — idempotent (overwrites).

Usage:
    python scripts/load_lint_fixtures.py [--wiki-dir WIKI_DIR]

Actions:
1. Copy eval/lint_fixtures/wiki/**/*.md into wiki/ (preserving subdirectory structure).
2. Append eval/lint_fixtures/log_entries.txt lines to wiki/log.md.
3. Touch eval/lint_fixtures/sources/aged_policy.md mtime to current time so C6 fires
   (the aged wiki page has updated: 2026-01-01 which will be older than the touched source).
4. Touch wiki/concepts/refund-policy-a.md mtime to current time so C9 fires for the
   live qa fixture qa-refund-window-003ghi (which has frontmatter.updated:
   2026-03-01T08:00:00Z and cites refund-policy-a#refund-timeline).

Revert:
    git checkout wiki/ && rm -f wiki/lint-report.md

Production wiki state is never polluted — eval/lint_fixtures/sources/ are not
copied to docs/; they are loaded into a wiki-side structure only.

The C6 (stale detection) check compares the source file's mtime against the wiki
page's frontmatter.updated timestamp. The loader touches aged_policy.md to now,
making it newer than the aged wiki page (updated: 2026-01-01).

The C9 (qa-staleness) check compares the wiki entity file's mtime against the
qa page's frontmatter.updated timestamp. The loader touches refund-policy-a.md
(in wiki/concepts/) to now, making it newer than the qa-refund-window page
(frontmatter.updated: 2026-03-01).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and fixture paths
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parent.parent  # scripts/ → repo root
_FIXTURES_DIR = _REPO_ROOT / "eval" / "lint_fixtures"
_FIXTURES_WIKI = _FIXTURES_DIR / "wiki"
_FIXTURES_LOG = _FIXTURES_DIR / "log_entries.txt"
_AGED_SOURCE = _FIXTURES_DIR / "sources" / "aged_policy.md"


def load_fixtures(wiki_dir: Path) -> None:
    """Load all lint fixtures into wiki_dir."""
    if not _FIXTURES_DIR.exists():
        print(f"ERROR: fixtures directory not found: {_FIXTURES_DIR}", file=sys.stderr)
        sys.exit(1)

    copied_pages: list[str] = []
    updated_pages: list[str] = []

    # Step 1: Copy fixture wiki pages into wiki_dir
    for fixture_page in sorted(_FIXTURES_WIKI.glob("**/*.md")):
        # Compute relative path under wiki/
        rel = fixture_page.relative_to(_FIXTURES_WIKI)
        dest = wiki_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        already_exists = dest.exists()
        shutil.copy2(str(fixture_page), str(dest))  # copy2 preserves metadata

        if already_exists:
            updated_pages.append(str(rel))
        else:
            copied_pages.append(str(rel))

    print(f"Pages copied (new): {len(copied_pages)}")
    for p in copied_pages:
        print(f"  + {p}")

    print(f"Pages overwritten (existing): {len(updated_pages)}")
    for p in updated_pages:
        print(f"  ~ {p}")

    # Step 2: Append log_entries.txt to wiki/log.md
    log_path = wiki_dir / "log.md"
    if _FIXTURES_LOG.exists():
        log_entries = _FIXTURES_LOG.read_text(encoding="utf-8")
        # Ensure log.md exists and has a trailing newline before appending
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8")
            if not existing.endswith("\n"):
                existing += "\n"
            log_path.write_text(existing + log_entries, encoding="utf-8")
            print(
                f"Log entries appended to {log_path} ({len(log_entries.splitlines())} lines)"
            )
        else:
            log_path.write_text(log_entries, encoding="utf-8")
            print(
                f"Log file created at {log_path} ({len(log_entries.splitlines())} lines)"
            )
    else:
        print(f"WARNING: log_entries.txt not found at {_FIXTURES_LOG}", file=sys.stderr)

    # Step 3: Touch aged_policy.md mtime to now so C6 (stale detection) fires
    # The aged wiki page has updated: 2026-01-01 which is before now.
    if _AGED_SOURCE.exists():
        now = time.time()
        import os

        os.utime(str(_AGED_SOURCE), (now, now))
        print("Touched aged_policy.md mtime to now (triggers C6 stale detection)")
    else:
        print(
            f"WARNING: aged_policy.md not found at {_AGED_SOURCE} — C6 check may not fire",
            file=sys.stderr,
        )

    # Step 4: Touch wiki/concepts/refund-policy-a.md mtime to now so C9
    # (qa-staleness) fires for qa-refund-window-003ghi (frontmatter.updated:
    # 2026-03-01T08:00:00Z). The qa page cites refund-policy-a#refund-timeline;
    # C9 compares the entity file's mtime against the qa page's frontmatter.updated.
    import os as _os

    refund_entity = wiki_dir / "concepts" / "refund-policy-a.md"
    if refund_entity.exists():
        now = time.time()
        _os.utime(str(refund_entity), (now, now))
        print(
            "Touched wiki/concepts/refund-policy-a.md mtime to now "
            "(triggers C9 qa-staleness for qa-refund-window-003ghi)"
        )
    else:
        print(
            f"WARNING: {refund_entity} not found — C9 check may not fire",
            file=sys.stderr,
        )

    print()
    print("Summary:")
    print(f"  Wiki pages loaded: {len(copied_pages) + len(updated_pages)}")
    print(
        f"  Log entries appended: {len(_FIXTURES_LOG.read_text(encoding='utf-8').splitlines()) if _FIXTURES_LOG.exists() else 0}"
    )
    print()
    print("To revert:")
    print("  git checkout wiki/")
    print("  rm -f wiki/lint-report.md")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load lint fixtures into wiki/ for demo or e2e testing."
    )
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=_REPO_ROOT / "wiki",
        help="Path to the wiki directory (default: <repo-root>/wiki)",
    )
    args = parser.parse_args()

    wiki_dir = args.wiki_dir.resolve()
    if not wiki_dir.exists():
        print(f"Creating wiki directory: {wiki_dir}")
        wiki_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading lint fixtures into: {wiki_dir}")
    print()

    load_fixtures(wiki_dir)


if __name__ == "__main__":
    main()
