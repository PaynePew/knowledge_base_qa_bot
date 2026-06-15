"""Deep module per Ousterhout. Public surface: ``parse_markdown``, ``split_frontmatter``, ``build_index``, ``load_index_json``, ``search``, ``slugify``, ``Section`` (dataclass), plus the module-level ``sections`` list (read by ``retrieval.py``).

Markdown Section Index builder.

Parses Markdown files under SOURCE_DIRS into Sections, builds a BM25 inverted
index in memory, and persists it as pretty-printed JSON to .kb/index.json.

The parse_markdown function follows the 10-rule body-bearing spec documented
in its docstring below.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
from ._paths import DOCS_DIR, INDEX_PATH, WIKI_DIR
from .atomic import write_text_atomic

# ADR-0006 (W1): build_index scans only the curated wiki subdirs.
# Whitelist semantics — meta-files (wiki/index.md, wiki/log.md, wiki/hot.md,
# wiki/README.md, wiki/.archive/*) are excluded by construction because only
# explicit subdirectories appear here. Phase 6 Slice 6-1 appends WIKI_DIR/"qa"
# but gates entry on ``frontmatter.status == "live"`` via
# ``_passes_index_filter`` below — see PRD #78 Q1 "Two-stage curation lifecycle".
SOURCE_DIRS: list[Path] = [WIKI_DIR / "entities", WIKI_DIR / "concepts", WIKI_DIR / "qa"]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}

# Thread-safety: callers hold _index_lock when swapping the sections list.
_index_lock = threading.Lock()

# Module-level snapshot of the last wiki index outcome from build_index().
# Populated by build_index() after calling write_wiki_index(). The route layer
# reads this instead of a return-value extension so existing test signatures
# for build_index() (tuple[int, int]) remain unchanged.
# Format: (wiki_written: bool, wiki_path: Path | None, wiki_error: str | None)
last_wiki_index_outcome: tuple[bool, Path | None, str | None] = (False, None, None)


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
        return asdict(self)


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
    """Lowercase ASCII chars, preserve CJK/Unicode letters, replace separators with hyphens.

    Follows the GitHub/Obsidian Unicode anchor convention (Phase 16):
    - ASCII letters are lowercased.
    - CJK characters (and other Unicode letters) are kept verbatim.
    - Runs of characters that are neither ASCII alphanumeric nor Unicode
      letters are collapsed to a single hyphen.
    - Leading/trailing hyphens are stripped.
    - When nothing slug-able remains, returns "section".

    Pure-ASCII input produces byte-identical output to the pre-Phase-16
    implementation (``re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")``).
    """
    # Build slug character by character:
    # - ASCII letter/digit → keep (lowercased)
    # - Unicode letter outside ASCII range → keep verbatim (CJK, etc.)
    # - Anything else → treated as a separator
    parts: list[str] = []
    sep_pending = False
    for ch in text:
        if ch.isalpha():
            if sep_pending and parts:
                parts.append("-")
                sep_pending = False
            parts.append(ch.lower() if ch.isascii() else ch)
        elif ch.isdigit() and ch.isascii():
            if sep_pending and parts:
                parts.append("-")
                sep_pending = False
            parts.append(ch)
        else:
            # Separator: emit a hyphen later (only if we have content before it)
            if parts:
                sep_pending = True
    slug = "".join(parts)
    return slug or "section"


# CJK Unified Ideographs range (most common block; covers Traditional/Simplified Chinese).
# Extension A/B and Compatibility Ideographs are handled by explicit codepoint
# ranges in _is_cjk() below (no unicodedata dependency).
_CJK_RANGE_START = ord("一")  # 一
_CJK_RANGE_END = ord("鿿")  # 鿿


def _is_cjk(ch: str) -> bool:
    """Return True when ``ch`` is a CJK Unified Ideograph or Extension."""
    cp = ord(ch)
    # CJK Unified Ideographs: U+4E00–U+9FFF
    if _CJK_RANGE_START <= cp <= _CJK_RANGE_END:
        return True
    # CJK Extension A: U+3400–U+4DBF
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Extension B: U+20000–U+2A6DF (supplementary plane)
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs: U+F900–U+FAFF
    return 0xF900 <= cp <= 0xFAFF


# Fraction of "letter" characters that must be CJK for a text to count as "zh".
# Letter characters = CJK ideographs + ASCII/Unicode alphabetic chars (the
# language-bearing signal); digits, whitespace, and punctuation are ignored so
# a Chinese sentence ending in "7天" or an English one with a stray "退" is
# classified by its dominant script, not diluted by neutral characters. The gate
# is deliberately low (a single CJK character among otherwise-Latin letters is
# rare in this corpus and almost always means the text is Chinese) but kept off
# zero so an all-Latin string with one accidental ideograph still reads "en"
# only when CJK is genuinely the minority.
_ZH_RATIO_THRESHOLD = 0.20

# The defined default for text that carries no language signal (empty,
# whitespace-only, or digits/symbols only — no CJK and no alphabetic letters).
# English is the fail-closed default for the bilingual demo (PRD #284): it is
# the larger corpus, and an untagged Chunk/Section defaulting to "en" never
# strips Chinese coverage that does not exist yet.
_DEFAULT_LANG = "en"

# Metadata key under which the index-time, content-derived language tag is
# stored on every Section (issue #285). It is a SYSTEM-INJECTED tag, distinct
# from author-written YAML frontmatter — consumers that reason about "did this
# page have frontmatter?" (e.g. the qa-status filter) must exclude this key.
LANG_METADATA_KEY = "lang"


def detect_lang(text: str) -> str:
    """Classify ``text`` as ``"zh"`` (Chinese) or ``"en"`` (English) by CJK ratio.

    The single, stable language-classification interface (issue #285, PRD #284).
    Consolidates the scattered "is this CJK?" logic — ``_is_cjk`` (ADR-0014
    bigram tokenisation) and ``retrieval._is_cjk_query`` (#261 threshold routing)
    — into one tested unit consumed by both index-time ``lang`` tagging and
    query-time routing, so the two can never drift.

    Decision rule (PRD #284 tie-break): compute the CJK character ratio over the
    *letter* characters only (CJK ideographs + alphabetic chars). Neutral
    characters — whitespace, digits, punctuation — are excluded so they neither
    inflate nor dilute the signal. The text is ``"zh"`` when that ratio crosses
    ``_ZH_RATIO_THRESHOLD``, else ``"en"``. Mixed text therefore resolves to the
    dominant script by ratio.

    Default: text with no letter characters at all (empty, whitespace-only,
    digits/symbols only) has no language signal and returns ``_DEFAULT_LANG``
    (``"en"``) — the defined fail-closed default.

    Pure function — no I/O, no mutation, deterministic for a given input. Derives
    its decision from content only; callers must pass content (never a filename
    or folder) per PRD #284 "Routing is metadata-driven, not folder-driven".
    """
    cjk = 0
    letters = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
            letters += 1
        elif ch.isalpha():
            letters += 1
    if letters == 0:
        # No language-bearing characters → no signal → defined default.
        return _DEFAULT_LANG
    return "zh" if cjk / letters >= _ZH_RATIO_THRESHOLD else "en"


def _section_metadata(frontmatter: dict, heading: str, content: str) -> dict:
    """Return a per-Section metadata dict carrying the content-derived ``lang``.

    Issue #285: every Section is tagged with ``lang`` (``"zh"``/``"en"``) so
    retrieval can later filter by language without a folder convention. The tag
    is derived from CONTENT, never the filename or folder (PRD #284): we classify
    the Section body, falling back to the heading text only when the body is
    empty (heading-only leaf, Rule 8) so an empty-body Section is still tagged by
    its own heading rather than left untagged.

    The frontmatter dict is copied first so each Section owns an independent
    metadata mapping (the prior ``dict(metadata)`` contract is preserved). A
    ``lang`` key already present in frontmatter is overwritten by the detected
    value — content is the source of truth, not a hand-authored hint.
    """
    meta = dict(frontmatter)
    # Body is the primary signal; fall back to the heading for empty-body leaves
    # (Rule 8). The heading here is a real Markdown heading at every call site
    # that owns body text — the only heading-without-body case is a leaf, whose
    # heading IS its content. Callers must never pass a filename as ``heading``.
    signal = content if content.strip() else heading
    meta[LANG_METADATA_KEY] = detect_lang(signal)
    return meta


def _bigrams(run: str) -> list[str]:
    """Produce sliding character bigrams from a CJK run.

    For a run of length 1 (single CJK character), return the character itself
    as a unigram fallback so single-character queries are never silently dropped.
    """
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def tokenize(text: str) -> list[str]:
    """Split text into tokens, removing stop words.

    Language-agnostic strategy (Phase 16):
    - CJK runs tokenise as sliding character bigrams (unigram fallback for
      length-1 runs). CJK bigrams are never filtered by STOP_WORDS (which
      contains only ASCII words).
    - All other text (Latin, digits, punctuation) tokenises via
      ``TOKEN_RE.findall(text.lower())`` with STOP_WORDS removal AND a length-1
      Latin-token filter (issue #252): single letters — chiefly possessive /
      contraction clitics ("What's" -> "s", "don't" -> "t") — carry no retrieval
      signal and produced spurious BM25 hits. Single DIGITS are preserved.
    - CJK logic only triggers on codepoints > 127, so CJK bigrams and the
      single-CJK-char unigram fallback are unaffected by the length-1 filter.
    """
    tokens: list[str] = []
    # Walk through the text, extracting CJK runs and non-CJK segments
    # alternately so each is handled by its own rule.
    current_cjk: list[str] = []
    non_cjk_buf: list[str] = []

    def _flush_cjk() -> None:
        if current_cjk:
            tokens.extend(_bigrams("".join(current_cjk)))
            current_cjk.clear()

    def _flush_non_cjk() -> None:
        if non_cjk_buf:
            segment = "".join(non_cjk_buf)
            # Drop length-1 Latin tokens (issue #252): a possessive/contraction
            # clitic ("What's" -> "s", "don't" -> "t") and stray single letters
            # carry no retrieval signal and create spurious BM25 hits. Single
            # DIGITS are kept (quantities/years matter), so filter only len-1
            # alphabetic tokens, not all len-1 tokens.
            tokens.extend(
                t
                for t in TOKEN_RE.findall(segment.lower())
                if t not in STOP_WORDS and not (len(t) == 1 and t.isalpha())
            )
            non_cjk_buf.clear()

    for ch in text:
        if _is_cjk(ch):
            _flush_non_cjk()
            current_cjk.append(ch)
        else:
            _flush_cjk()
            non_cjk_buf.append(ch)

    _flush_cjk()
    _flush_non_cjk()
    return tokens


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading YAML frontmatter block from a Markdown document.

    Reusable form of the frontmatter-detection convention applied inline by
    ``parse_markdown`` Rule 2. Used by the ingest classifier (issue #106) so the
    raw Source text fed to the LLM excludes provenance frontmatter that
    importer.py writes (``imported_from``/``original_format``/``imported_at``/
    ``content_sha256``). Recognises that on-disk format: a document beginning
    with ``---\\n``, its closing fence ``\\n---\\n``, and the body following.

    Returns ``(metadata, body)``:
    - metadata: parsed YAML mapping (``{}`` when absent, malformed, or PyYAML is
      unavailable).
    - body: the document text with the leading frontmatter block removed
      (byte-identical to the input when no frontmatter is present, so the
      no-frontmatter path is unaffected).

    Never raises: missing closing fence, malformed YAML, and absent PyYAML all
    fall back to ``({}, text)`` and emit a ``parse_warning`` (consistent with
    parse_markdown's fail-soft handling).
    """
    from .logger import log_event

    if not text.startswith("---\n"):
        return {}, text

    try:
        import yaml  # optional; PyYAML is not yet a hard dep (Wiki layer territory)

        end = text.index("\n---\n", 4)  # closing fence
        metadata = yaml.safe_load(text[4:end]) or {}
        return metadata, text[end + 5 :]  # skip \n---\n
    except ImportError:
        log_event(
            "parse_warning",
            "frontmatter present but PyYAML is not installed; treating as no frontmatter",
        )
        return {}, text
    except (yaml.YAMLError, ValueError) as exc:
        # yaml.YAMLError → malformed YAML; ValueError → missing closing fence
        log_event(
            "parse_warning",
            f"frontmatter split failed: {type(exc).__name__}",
        )
        return {}, text


# ---------------------------------------------------------------------------
# Core parser — 10-rule body-bearing spec
# ---------------------------------------------------------------------------


def parse_markdown(path: Path, source_id: str | None = None) -> list[Section]:
    """Parse one Markdown file into Sections under the body-bearing rule.

    ``source_id`` overrides the filename as the prefix in ``Section.id`` and
    the value of ``Section.file``. Pass ``path.stem`` (bare slug, no ``.md``)
    for wiki-derived pages so citations use the slug-based addressing scheme
    (e.g. ``refund-policy#cancellation-window`` instead of
    ``refund-policy.md#cancellation-window``). When omitted, the full filename
    (``path.name``) is used — preserving the existing behaviour for docs/.

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
    7.  A Source with zero headings produces a single Section: `id=source_id`
        (or `id=filename` when source_id is None) (no `#anchor`),
        `heading=source_id`, `heading_path=[source_id]`,
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
    # source_prefix is what goes into Section.id (prefix before '#') and Section.file.
    # When source_id is provided (wiki-derived pages), use the bare slug so citations
    # render as 'refund-policy#cancellation-window' instead of
    # 'refund-policy.md#cancellation-window'.
    source_prefix = source_id if source_id is not None else filename
    raw = path.read_text(encoding="utf-8")

    # Rule 2: YAML frontmatter
    metadata: dict = {}
    body = raw
    if raw.startswith("---\n"):
        try:
            import yaml  # optional; PyYAML is not yet a hard dep (Wiki layer territory)

            end = raw.index("\n---\n", 4)  # closing fence
            fm_text = raw[4:end]
            metadata = yaml.safe_load(fm_text) or {}
            body = raw[end + 5 :]  # skip \n---\n
        except ImportError:
            log_event(
                "parse_warning",
                f"frontmatter present in {filename} but PyYAML is not installed; treating as no frontmatter",
            )
            body = raw
        except (yaml.YAMLError, ValueError) as exc:
            # yaml.YAMLError → malformed YAML; ValueError → missing closing fence
            log_event(
                "parse_warning",
                f"frontmatter parse failed in {filename}: {type(exc).__name__}",
            )
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
            return f"{source_prefix}#{slug}"
        used_slugs[slug] += 1
        return f"{source_prefix}#{slug}-{used_slugs[slug]}"

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
                    file=source_prefix,
                    heading=heading_text,
                    heading_path=entry["path"],
                    content=content,
                    tokens=tokens,
                    metadata=_section_metadata(metadata, heading_text, content),
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
            stack.append(
                {
                    "depth": new_depth,
                    "heading": heading_text,
                    "body_lines": [],
                    "has_child": False,
                    "path": ancestor_path + [heading_text],
                }
            )
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
        whole_body = body.strip()
        result.append(
            Section(
                id=source_prefix,
                file=source_prefix,
                heading=source_prefix,
                heading_path=[source_prefix],
                content=whole_body,
                tokens=tokenize(body),
                # Rule 7 no-heading Section: classify the body only. Pass an
                # empty heading fallback so a body-less file defaults to "en"
                # via detect_lang without the filename (source_prefix) leaking
                # into the language decision (PRD #284 content-not-filename).
                metadata=_section_metadata(metadata, "", whole_body),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Phase 6 Slice 6-1: qa-filter (PRD #78 Q1 + Q8d)
# ---------------------------------------------------------------------------


def _passes_index_filter(md_file: Path, page_sections: list[Section]) -> bool:
    """Return True iff ``md_file``'s sections may join the BM25 corpus.

    Phase 6 Two-stage curation gate (PRD #78 Q1): pages under ``wiki/qa/`` are
    admitted only when ``frontmatter.status == "live"``. ``status == "draft"``
    is the healthy filed-but-not-promoted state — skipped silently. Every
    other value (capital-L typos, missing key, forward-compat values like
    ``stale``/``superseded``) is treated as a curator-typo orphan and skipped
    with a ``qa_invalid_status`` log entry — the indexer-layer member of the
    three-layer orphan-visibility defence (PRD #78 §"Orphan-visibility
    three-layer defence").

    Entity and concept pages bypass the filter entirely so Phase 3 behaviour
    is preserved without regression.

    ``page_sections`` is the result of ``parse_markdown`` for ``md_file``.
    Frontmatter is read off ``page_sections[0].metadata`` (every Section
    carries a copy of the file's frontmatter dict). An empty ``page_sections``
    list short-circuits to ``False`` without emitting ``qa_invalid_status`` —
    ``parse_markdown`` has already emitted a ``parse_warning`` for that file,
    and double-logging is exactly the noise this layer is trying to prevent.
    """
    # Empty parse result: parse_warning already covers it, do not double-log.
    if not page_sections:
        return False

    # Non-qa pages (entity, concept) pass through unchanged.
    if md_file.parent.name != "qa":
        return True

    # qa page: gate on frontmatter.status. Exclude the system-injected language
    # tag (issue #285) so this emptiness check still asks "did the author write
    # any YAML frontmatter?" — every Section now carries a ``lang`` key, which
    # must not be mistaken for real frontmatter content.
    metadata = page_sections[0].metadata
    author_frontmatter = {k: v for k, v in metadata.items() if k != LANG_METADATA_KEY}
    # Empty / absent metadata (e.g. frontmatter parse failed, or no frontmatter
    # at all on a qa page) — parse_markdown already emitted parse_warning when
    # the YAML was malformed. Skip silently to avoid the double-log noise that
    # PRD #78 §"Orphan-visibility three-layer defence" explicitly calls out.
    if not author_frontmatter:
        return False
    status = author_frontmatter.get("status")

    if status == "live":
        return True
    if status == "draft":
        # Healthy intermediate state in two-stage curation; silent skip.
        return False

    # Any other value (capital-L typo, missing key, stale, superseded, ...)
    # is an orphan from the indexer's perspective. Log + skip.
    from .logger import log_event

    log_event(
        "qa_invalid_status",
        f"file={md_file.name} status={status!r}",
    )
    return False


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
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0
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

    # Atomic write via shared helper (CODING_STANDARD §2.6, issue #211).
    # Serialise to a string first so write_text_atomic can force LF newlines.
    index_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    write_text_atomic(index_path, index_json)


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

    After writing the Section Index JSON, calls write_wiki_index(sections) as a
    best-effort side effect. The wiki write outcome (success or failure) is stored
    in the module-level ``last_wiki_index_outcome`` variable so the route layer
    can surface it in ``IndexResponse`` without any change to this function's
    return signature. Wiki write failure is non-blocking — the function returns
    normally and the Section Index is still served. On failure a
    ``wiki_index_error`` log entry is emitted.

    Design choice: module-level snapshot variable over return-value extension.
    Rationale: existing test signatures for build_index() expect ``tuple[int, int]``
    and are not touched by this slice. The route reads ``last_wiki_index_outcome``
    directly after calling build_index().

    Returns (files_indexed, sections_indexed).
    """
    global sections, doc_freq, avg_doc_len, files_indexed, last_wiki_index_outcome
    from . import wiki_index as _wiki_index_module
    from .logger import log_event

    # Determine which directories to scan.
    # If caller passes a non-default docs_dir, use just that (test isolation).
    # In production the default hits SOURCE_DIRS.
    scan_dirs = [docs_dir] if docs_dir is not DOCS_DIR else SOURCE_DIRS

    new_sections: list[Section] = []
    # Use bare slug (stem, no .md) as source_id only for the production SOURCE_DIRS
    # path so wiki-derived Sections cite as 'refund-policy#heading' instead of
    # 'refund-policy.md#heading'. When caller passes an explicit docs_dir (test
    # isolation), scan_dirs == [docs_dir] and source_id is left as None, preserving
    # the existing filename-based addressing for docs/ content.
    use_slug_ids = docs_dir is DOCS_DIR
    for source_dir in scan_dirs:
        for md_file in sorted(source_dir.glob("**/*.md")):
            sid = md_file.stem if use_slug_ids else None
            page_sections = parse_markdown(md_file, source_id=sid)
            # Phase 6 Slice 6-1: qa-filter gates wiki/qa/ pages on
            # frontmatter.status == "live"; non-qa pages pass through.
            if not _passes_index_filter(md_file, page_sections):
                continue
            new_sections.extend(page_sections)

    with _index_lock:
        sections = new_sections
        rebuild_stats()

    write_index_json()

    log_event(
        "index_built",
        f"files={files_indexed} sections={len(sections)}",
    )

    # ADR-0006: emit wiki_layer_empty when both whitelisted wiki subdirs scan to
    # zero sections — distinct ops signal from routine index_missing so /lint
    # (Phase 5) can tell "system deployed but never ingested" apart from a
    # normal cannot-confirm. Only fires when using the default SOURCE_DIRS
    # (production path); skipped when caller passes an explicit docs_dir
    # (test-isolation path).
    if docs_dir is DOCS_DIR and len(sections) == 0:
        log_event(
            "wiki_layer_empty",
            "entities=0 concepts=0",
        )

    # Best-effort wiki index write — non-blocking; failure stored for route layer.
    # Called via module reference so tests can monkeypatch _wiki_index_module.write_wiki_index.
    wiki_written, wiki_path, wiki_error = _wiki_index_module.write_wiki_index(sections)
    last_wiki_index_outcome = (wiki_written, wiki_path, wiki_error)
    if not wiki_written:
        # Best-effort log — if the filesystem is the problem, this may also fail;
        # do not catch or propagate the secondary failure.
        log_event("wiki_index_error", f"reason={wiki_error}")

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


def _section_lang(section: Section) -> str:
    """Return a Section's language, preferring its index-time ``lang`` tag.

    Every Section built since #285 carries ``metadata['lang']`` (``"zh"``/``"en"``).
    For a legacy index built before that tag existed, fall back to classifying the
    Section content with the same ``detect_lang`` helper — so the language filter
    works on a not-yet-rebuilt corpus without silently dropping every Section.
    Content is the source of truth either way (PRD #284), so the fallback agrees
    with what a rebuild would tag.
    """
    tagged = (section.metadata or {}).get(LANG_METADATA_KEY)
    if tagged in ("zh", "en"):
        return tagged
    return detect_lang(section.content)


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    """Return the top-``k`` BM25 Sections in the QUERY's language.

    Language-filtered retrieval (#287, PRD #284): a Chinese query is scored only
    against ``zh``-tagged Sections and an English query only against ``en``-tagged
    ones, using the consolidated ``detect_lang`` query-language predicate (#285) so
    query-time routing and index-time ``lang`` tagging share one classifier and
    never drift.

    This makes EXPLICIT the routing that CJK-bigram tokenisation (ADR-0014) already
    does implicitly (English tokens never match Chinese bigrams), so it does not
    regress same-language answers. Its load-bearing effect is closing the residual
    cross-language leak the implicit routing misses: a query and a wrong-language
    Section can still share an ASCII token (a brand name, a number, a code), which
    under plain BM25 lets the wrong-language Section match — and sometimes out-rank
    — the right one. Filtering by language is belt-and-suspenders for the BM25 stack
    and the essential gate for the RAG stack (PRD #284).
    """
    query_lang = detect_lang(query)
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
        if _section_lang(section) == query_lang
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]


def expand_to_pages(hits: list[Section]) -> list[Section]:
    """Expand BM25 hits to full parent pages.

    Pure function over ``indexer.sections``. No side-effects, no I/O.

    Contract:
    - Input: BM25 hits at Section granularity (Section list, no scores).
    - Output: hits ∪ all sibling Sections of their parent pages.
    - Page key: ``Section.file`` (bare slug under A2, post-#53).
    - Page ordering: the page whose top hit ranks highest (i.e. appears
      earliest in ``hits``) comes first in the output.
    - Section ordering within a page: document order as determined by
      the order of entries in ``indexer.sections``.
    - Deduplication: each parent page is expanded exactly once, even when
      multiple BM25 hits belong to the same page.

    Returns an empty list when ``hits`` is empty.
    """
    if not hits:
        return []

    # Determine the rank of each page by the position of its first (best) hit
    # in the input hits list. Lower index = higher rank = comes first.
    page_rank: dict[str, int] = {}
    for rank, sec in enumerate(hits):
        if sec.file not in page_rank:
            page_rank[sec.file] = rank

    # Collect all sections of each hit page from the module-level sections list,
    # preserving document order (= position in indexer.sections).
    pages: dict[str, list[Section]] = {file: [] for file in page_rank}
    for sec in sections:
        if sec.file in pages:
            pages[sec.file].append(sec)

    # Sort pages by their rank (best hit position), then flatten in document order.
    sorted_files = sorted(page_rank.keys(), key=lambda f: page_rank[f])
    result: list[Section] = []
    for file in sorted_files:
        result.extend(pages[file])

    return result
