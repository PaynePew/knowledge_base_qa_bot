"""Markdown Section Index builder.

Parses Markdown files under SOURCE_DIRS into Sections, builds a BM25 inverted
index in memory, and persists it as pretty-printed JSON to .kb/index.json.

The parse_markdown function follows the 10-rule body-bearing spec documented
in its docstring below.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
WIKI_DIR = Path(__file__).resolve().parents[2] / "wiki"
INDEX_PATH = Path(__file__).resolve().parents[2] / ".kb" / "index.json"

# ADR-0003: build_index iterates this list so adding WIKI_DIR needs no
# signature change.
SOURCE_DIRS: list[Path] = [DOCS_DIR]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}

# Thread-safety: callers hold _index_lock when swapping the sections list.
_index_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]
    metadata: dict = field(default_factory=dict)  # YAML frontmatter (future use)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# In-memory index state
# ---------------------------------------------------------------------------

sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Lowercase, replace non-alphanumeric runs with hyphens, strip edges."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens, removing stop words."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Core parser — 10-rule body-bearing spec
# ---------------------------------------------------------------------------


def parse_markdown(path: Path) -> list[Section]:
    """Parse one Markdown file into Sections under the body-bearing rule.

    See CONTEXT.md > Section for the formal definition. The 10-rule spec:

    1.  Read the file as UTF-8.
    2.  If the file starts with `---\\n`, strip and parse the YAML frontmatter
        into a dict. Attach this dict to every Section's `metadata` field.
        Do NOT tokenize frontmatter values into BM25 tokens.
    3.  Scan the remaining body line by line, maintaining `in_fence: bool`.
        Toggle `in_fence` whenever a line starts with three backticks. While
        `in_fence` is true, treat every line as content — do NOT match
        HEADING_RE against fenced code (so `# bash comment` inside a code
        block is not treated as a heading).
    4.  Outside fences, match HEADING_RE. Use a stack to track the current
        heading path. When a heading at depth d arrives, pop the stack until
        the top has depth < d (those headings are "closed" and emitted as
        Sections if they qualify under rule 5). Then push the new heading.
    5.  A heading becomes a Section when either:
            (i)  It is a leaf — from its push to its pop, no deeper heading
                 was ever pushed on top of it; OR
            (ii) It is body-bearing — between its push and the first deeper
                 heading pushed on top of it, the body content accumulated
                 directly under it is not whitespace-only.
        In case (ii) the Section's content is only the body owned directly
        by this heading, NOT the recursive content of its children.
    6.  Emit a `log_event("parse_warning", ...)` whenever a non-leaf heading
        has only whitespace body and therefore produces no Section (this is
        normal for h1 file titles, but worth logging at startup).
    7.  A Source with zero headings produces a single Section: `id=filename`
        (no `#anchor`), `heading=filename`, `heading_path=[filename]`,
        `content=` full file body.
    8.  An empty-body leaf (heading present, body whitespace-only) is still
        emitted as a Section. Its `content` is `""`; its `tokens` come from
        the heading text alone. BM25 will rank it low unless the query
        matches the heading directly, which is the desired behavior.
    9.  Heading slug collisions inside the same Source: append `-2`, `-3`, …
        suffixes. Never silently overwrite a previously emitted Section.
    10. `tokens` is the concatenation of (a) lowercase alphanumeric tokens
        from the heading text and (b) the same for the body content, with
        STOP_WORDS removed. The same tokenization applies to query strings
        at retrieval time.
    """
    # Import here to avoid circular dependency at module level; logger imports
    # nothing from indexer.
    from .logger import log_event

    filename = path.name
    raw = path.read_text(encoding="utf-8")

    # Rule 2: YAML frontmatter
    metadata: dict = {}
    body = raw
    if raw.startswith("---\n"):
        try:
            import yaml  # optional; graceful fallback if not installed
            end = raw.index("\n---\n", 4)
            fm_text = raw[4:end]
            metadata = yaml.safe_load(fm_text) or {}
            body = raw[end + 5:]  # skip \n---\n
        except (ValueError, ImportError, Exception):
            # If YAML parse fails or PyYAML not installed, treat as no frontmatter
            body = raw

    lines = body.splitlines(keepends=True)

    # Stack entries: dict with keys:
    #   depth       int             — '#' count
    #   heading     str             — raw heading text
    #   body_lines  list[str]       — lines accumulated directly under this heading
    #   has_child   bool            — True once a deeper heading was pushed on top
    #   path        list[str]       — full heading_path captured at push time
    stack: list[dict] = []
    result: list[Section] = []
    used_slugs: dict[str, int] = {}  # slug → next suffix counter
    in_fence = False
    found_any_heading = False

    def _make_id(slug: str) -> str:
        """Apply collision-safe suffix and track usage."""
        if slug not in used_slugs:
            used_slugs[slug] = 1
            return f"{filename}#{slug}"
        used_slugs[slug] += 1
        return f"{filename}#{slug}-{used_slugs[slug]}"

    def _emit(entry: dict) -> None:
        """Emit a Section for a closed heading if it qualifies under rule 5."""
        heading_text = entry["heading"]
        raw_body = "".join(entry["body_lines"])
        body_stripped = raw_body.strip()
        is_leaf = not entry["has_child"]

        if is_leaf or body_stripped:
            # Rule 5(i) leaf, or rule 5(ii) body-bearing intermediate
            slug = slugify(heading_text)
            sec_id = _make_id(slug)
            content = body_stripped  # Rule 8: empty-body leaf has content=""
            tokens = tokenize(heading_text) + tokenize(raw_body)

            result.append(
                Section(
                    id=sec_id,
                    file=filename,
                    heading=heading_text,
                    heading_path=entry["path"],
                    content=content,
                    tokens=tokens,
                    metadata=dict(metadata),
                )
            )
        else:
            # Rule 6: non-leaf heading with whitespace-only body → log_event
            log_event(
                "parse_warning",
                f"non-leaf heading with no body in {filename}: '{heading_text}'",
            )

    for line in lines:
        # Rule 3: fence toggle (detect triple-backtick at start of line)
        stripped = line.rstrip("\n")
        if stripped.startswith("```"):
            in_fence = not in_fence
            # Add the fence line itself to the current heading's body
            if stack:
                stack[-1]["body_lines"].append(line)
            continue

        if in_fence:
            # Inside a fence: treat as content, not a heading
            if stack:
                stack[-1]["body_lines"].append(line)
            continue

        # Rule 4: outside fences, try to match a heading
        m = HEADING_RE.match(stripped)
        if m:
            found_any_heading = True
            new_depth = len(m.group(1))
            heading_text = m.group(2)

            # Pop headings at depth >= new_depth, emitting them
            while stack and stack[-1]["depth"] >= new_depth:
                entry = stack.pop()
                _emit(entry)

            # Mark the current top of stack (if any) as having a child
            if stack:
                stack[-1]["has_child"] = True

            # Compute heading_path at push time: ancestors + this heading
            ancestor_path = [e["heading"] for e in stack]
            # Push the new heading
            stack.append({
                "depth": new_depth,
                "heading": heading_text,
                "body_lines": [],
                "has_child": False,
                "path": ancestor_path + [heading_text],
            })
        else:
            # Content line — accumulate on current heading's body
            if stack:
                stack[-1]["body_lines"].append(line)

    # Pop remaining headings off the stack (end of file)
    while stack:
        entry = stack.pop()
        _emit(entry)

    # Rule 7: no headings found → single Section for the whole file
    if not found_any_heading:
        result.append(
            Section(
                id=filename,
                file=filename,
                heading=filename,
                heading_path=[filename],
                content=body.strip(),
                tokens=tokenize(body),
                metadata=dict(metadata),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Index build, persist, load
# ---------------------------------------------------------------------------


def rebuild_stats() -> None:
    """Rebuild doc_freq, avg_doc_len, and files_indexed from the in-memory sections."""
    global doc_freq, avg_doc_len, files_indexed
    doc_freq = Counter()
    for sec in sections:
        for tok in set(sec.tokens):
            doc_freq[tok] += 1
    avg_doc_len = (
        sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0
    )
    files_indexed = len({s.file for s in sections})


def write_index_json(index_path: Path | None = None) -> None:
    """Persist the section index to .kb/index.json atomically.

    Writes {"sections": [...], "stats": {...}} as pretty-printed JSON.
    Uses a temp file + os.replace for atomicity.
    """
    if index_path is None:
        index_path = INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sections": [s.to_dict() for s in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
            "doc_freq": dict(doc_freq),
        },
    }

    # Atomic write: write to a sibling tmp file then os.replace
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=index_path.parent, suffix=".tmp", prefix="index_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path_str, index_path)
    except Exception:
        # Clean up tmp file on failure
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def load_index_json(index_path: Path | None = None) -> tuple[int, int]:
    """Load .kb/index.json into the in-memory sections list.

    Returns (files_indexed, sections_indexed). Returns (0, 0) if the file
    does not exist.

    Raises json.JSONDecodeError (or any other parse exception) on a corrupt
    index file so the server fails fast rather than silently serving wrong data.

    On a successful load, emits an ``index_loaded | files=N sections=M`` entry
    to wiki/log.md so the log records server boot + index reload across restarts.
    """
    from .logger import log_event

    global sections

    if index_path is None:
        index_path = INDEX_PATH

    if not index_path.exists():
        return 0, 0

    raw = index_path.read_text(encoding="utf-8")
    # Let json.JSONDecodeError propagate — corrupt index is a fail-fast condition.
    payload = json.loads(raw)

    with _index_lock:
        sections = [
            Section(
                id=item["id"],
                file=item["file"],
                heading=item["heading"],
                heading_path=item["heading_path"],
                content=item["content"],
                tokens=item["tokens"],
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("sections", [])
        ]
        rebuild_stats()

    log_event(
        "index_loaded",
        f"files={files_indexed} sections={len(sections)}",
    )

    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build an in-memory section index from SOURCE_DIRS.

    Per ADR-0003, iterates SOURCE_DIRS so adding WIKI_DIR requires no
    signature change. When docs_dir is provided (non-default), only that
    directory is indexed (used in tests).

    Returns (files_indexed, sections_indexed).
    """
    global sections, doc_freq, avg_doc_len, files_indexed
    from .logger import log_event

    # Determine which directories to scan.
    # If caller passes a non-default docs_dir, use just that (test isolation).
    # In production the default hits SOURCE_DIRS.
    if docs_dir is not DOCS_DIR:
        scan_dirs = [docs_dir]
    else:
        scan_dirs = SOURCE_DIRS

    new_sections: list[Section] = []
    for source_dir in scan_dirs:
        for md_file in sorted(source_dir.glob("*.md")):
            new_sections.extend(parse_markdown(md_file))

    with _index_lock:
        sections = new_sections
        rebuild_stats()

    write_index_json()

    log_event(
        "index_built",
        f"files={files_indexed} sections={len(sections)}",
    )

    return files_indexed, len(sections)


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------


def bm25_score(
    query_tokens: list[str],
    section: Section,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Score one section for the query using BM25."""
    if not sections:
        return 0.0
    n = len(sections)
    score = 0.0
    token_counts = Counter(section.tokens)
    doc_len = len(section.tokens)
    norm = 1 - b + b * (doc_len / avg_doc_len) if avg_doc_len > 0 else 1.0
    for tok in query_tokens:
        tf = token_counts.get(tok, 0)
        if tf == 0:
            continue
        df = doc_freq.get(tok, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
    # Small heading path boost
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    boost = sum(0.5 for tok in query_tokens if tok in heading_tokens)
    return score + boost


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
