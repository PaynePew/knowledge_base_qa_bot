"""Deep module per Ousterhout. Public surface: ``run_lint``, ``_check_c11_orphan``, ``check_full_orphan``, ``_check_c3_failed_grounding``, ``_check_c4a_slug_collision``, ``_check_c6_stale``, ``_check_c2_red_links``, ``_check_c1_coverage_gaps``, ``_canonicalise``, ``_candidate_pairs``, ``_judge_page_pair``, ``_check_c5_page_pair``, ``_load_wiki_pages``, ``get_lint_llm``, ``generate_reconcile_draft``, ``find_inbound_references``, ``generate_collision_merge_draft``, ``generate_collision_differentiate_draft``, ``_check_c8_promotion_candidates``, ``_check_c9_qa_staleness``, ``_check_c10_qa_schema_validity``, ``group_findings_by_axis``, ``LINT_CHECK_TAXONOMY``, ``LINT_AXIS_ORDER``, ``LINT_AXIS_LABEL_ZH``, ``remediation_for``, ``RemediationDescriptor``, ``RemediationAction``, ``DIFFERENTIATE_SENTINEL_TEMPLATE``.

Lint orchestrator — POST /lint health check for the wiki.

Provides ``run_lint(*, wiki_dir, docs_dir, log_path)`` which orchestrates the
lint checks and writes ``wiki/lint-report.md``.

Slice 5-1 scope
---------------
C11 (orphan pages) is wired.

Slice 5-2 scope
---------------
C3 (failed-grounding sweep) and C4-a (slug collision groups) are added.
Subsequent slices add the remaining checks without changing the orchestrator
contract or the continue-on-error semantics established here.

Slice 5-3 scope
---------------
C6 (mtime-based stale detection) and C2 (red link backlog) are wired.

Slice 5-4 scope
---------------
C1 (coverage gap aggregation from chat_fallback log) is wired.  Reads
``wiki/log.md`` line by line, clusters ``chat_fallback`` and
``chat_grounding_fallback`` entries by reason and canonical query key, and
surfaces the result as a curator-actionable coverage backlog.

Slice 5-5 scope
---------------
C5 (page-pair LLM contradiction detection) is wired.  Uses F1 ∪ F3 candidate
filter before calling the LLM via ``get_lint_llm()`` lazy singleton.
``OPENAI_LINT_MODEL`` env var with 3-layer fallback.  ``temperature=0`` for
determinism.  ``KB_LINT_BM25_TOP_K`` and ``KB_LINT_BM25_THRESHOLD`` control the
F3 filter.  Pairs are symmetrically canonicalised (sorted slug names) so each
pair is judged exactly once.  Continue-on-error: LLM errors mid-batch retain
prior findings.  Cost accounting via response metadata (best-effort token counts).

Slice 6-5 scope (Phase 5 amendment — PRD #78)
---------------------------------------------
Three new read-only checks scan ``wiki/qa/*.md`` and one modifier excludes
``type=qa`` pages from C5 pair generation:

- **C5 modifier** — ``_candidate_pairs`` filters ``frontmatter.type == "qa"``
  BEFORE F1/F3 candidate computation, preserving C5 LLM call budget and
  preventing trivially-true ``duplicate`` findings between qa pages and their
  source entities (PRD #78 Q1 + Q6).
- **C8 promotion candidates** — surfaces ``status: draft`` Filed Answers ranked
  by ``count`` desc / ``updated`` desc to ``lint-report.md`` §
  ``### C8 Promotion Candidates``. Capped by ``KB_LINT_PROMOTION_TOP_N`` env
  var (default 10). Read-only — the actual draft→live mutation is owned by
  Phase 6 ``POST /qa/{slug}/promote``.
- **C9 qa-staleness** — for each ``status: live`` Filed Answer, compares each
  cited entity file's mtime against ``frontmatter.updated``. Newer entities
  surface to ``### C9 Stale Filed Answers``. Closes Q6b "entity re-ingested,
  qa stranded" failure mode.
- **C10 qa-schema-validity** — sweeps ``wiki/qa/`` for invalid frontmatter
  (``status`` outside ``{live, draft, stale, superseded}``, missing/empty
  ``question``, ``type != "qa"``, missing/non-positive ``count``). Surfaces to
  ``### C10 Invalid qa Schema``. Closes Q8d "curator-typo orphan zombie"
  failure mode (third layer of the indexer-log + filing-refuse + lint
  defence stack).

Slice S1 scope (Lint Remediation tier-A — issue #361, ADR-0023)
-----------------------------------------------------------------
No new check. ``LINT_CHECK_TAXONOMY`` maps each of the ten wired checks to a
``{code, label, axis}`` (CONTEXT.md "Lint Axis": Freshness -> Coherence ->
Coverage -> Lifecycle) and ``group_findings_by_axis`` turns a run's
``LintFindings`` into that ordered structure. ``_render_report_markdown``
groups every check section under its axis heading and labels each check
section with its taxonomy code + label, so a later remediation slice (or the
CLI/MCP surfaces, per ADR-0017 interface parity) can reuse the same taxonomy.
Lint remains read-only — this is a report-layout change only.

Slice S3 scope (Lint Remediation tier-A — issue #363, ADR-0023)
-----------------------------------------------------------------
No new check, no new endpoint. ``remediation_for`` maps each of the ten
wired checks to a ``RemediationDescriptor`` (tier + executable actions),
reusing S1's ``LINT_CHECK_TAXONOMY`` codes as keys. The Operator Console is
its first consumer: Direct-tier findings (C6/C3 stale/failed-grounding,
C10 invalid-schema) render a per-row Remediation button wired to the
*existing* ``POST /ingest`` / ``DELETE /qa/{slug}`` endpoints; Authored-tier
findings (C5/C4/C2/C1) render a disabled tier-B affordance; deferred
findings (C9/C11) render neither (no lifecycle endpoint exists yet). Lint
itself is untouched — remediation is always a separate operation triggered
from the report, never a side-effect of ``run_lint()``.

Slice S5 scope (Lint Remediation tier-A — issue #365, ADR-0023)
-----------------------------------------------------------------
No new check, no new endpoint. Each ``LintCheckMeta`` entry in
``LINT_CHECK_TAXONOMY`` gains a ``label_zh`` field (the Traditional-Chinese
short label), and ``LINT_AXIS_LABEL_ZH`` maps each ``LINT_AXIS_ORDER``
identifier to its zh display string. This is the single source of truth for
the Operator Console's zh/en header toggle (structural chrome only — axis
headers, check labels, remediation button verbs, section chrome, empty
states); the dynamic per-finding ``suggested_action`` text stays English.
The axis identifiers themselves (``LINT_AXIS_ORDER`` values, dict keys)
stay English — only their *display* form is bilingual, so the written
report / CLI / MCP renderers (English-only, unchanged by this slice) keep
using the same stable keys.

All four amendments preserve PRD #65 Q3 read-only invariant — they read
frontmatter and write only ``lint-report.md``, never page frontmatter.

Slice S1 scope (Lint Remediation tier-B — issue #376, ADR-0028)
-----------------------------------------------------------------
No new check. ``generate_reconcile_draft`` adds a second C5-adjacent LLM
call site (page-pair *drafting*, distinct from ``_judge_page_pair``'s
*judging*) reusing the SAME ``get_lint_llm()`` lazy singleton — kept inside
this module rather than opening a second LLM-facing module for one call
site (ADR-0005 § "LLM-facing surface enumeration" already blesses
``lint.py``'s ``ChatOpenAI`` for "contradiction ... checks"). The actual
disk write-back, hash-based optimistic-concurrency check, and grounding
re-verification live in the new ``reconcile.py`` deep module (writes) —
this module stays read-only with respect to wiki page frontmatter, per the
invariant below; drafting a reconcile is not itself a mutation.

Slice S2 scope (Lint Remediation tier-B — issue #378, ADR-0028)
-----------------------------------------------------------------
No new check. C4's slug-collision groups gain both documented resolutions on
top of S1's two-phase machinery: ``generate_collision_merge_draft`` (merge
every group member into the unsuffixed base slug) and
``generate_collision_differentiate_draft`` (rewrite every group member in
place to be complementary) add a third and fourth C5-adjacent LLM call site,
reusing the SAME ``get_lint_llm()`` lazy singleton. ``find_inbound_references``
is read-only (no LLM call) and scans wiki links + qa citations for a single
slug — the C4 merge-apply reference guard (``reconcile.py``) uses it to
refuse deleting a variant that is still referenced. Disk write-back, hashing,
and grounding re-verification live in ``reconcile.py``, mirroring S1.

Slice S4 scope (Lint Remediation tier-B — issue #380, ADR-0026)
-----------------------------------------------------------------
No new check, no change to this module's checks or report rendering. The
new ``POST /qa/{slug}/refile`` endpoint (chained re-synthesis + grounding-
check + demote-in-place, implemented in ``qa.py``) is C9's remediation, so
``_REMEDIATION_TAXONOMY["C9"]`` flips from ``"deferred"`` to an Authored
descriptor carrying a ``refile`` action — the shared taxonomy this table
already is (S3) now tells every consumer (Console, and any future CLI/MCP
tier label) that C9 is remediable, matching C8's ``page_slug`` target-field
convention. This module has no write path of its own for it: the write,
the LLM-free-vs-LLM-based distinction, and the invariant ("a failed
re-ground writes nothing") all live in ``qa.refile``.

Slice S5 scope (Lint Remediation tier-B — issue #381, ADR-0025)
-----------------------------------------------------------------
No new check. C11's missing-citation scan is refactored into
``_orphan_predicate`` (shared helper) so both ``_check_c11_orphan`` (the bulk
sweep) and the new public ``check_full_orphan`` (the ``DELETE /pages/{slug}``
re-verification entry point, called from the new ``pages.py`` deep module)
compute the SAME full/partial split — ``OrphanPageFinding`` gains a ``full``
field (CONTEXT.md "Orphan Page": every citation missing vs some surviving).
``_REMEDIATION_TAXONOMY["C11"]`` flips from ``"deferred"`` to a Confirmed
descriptor carrying a ``delete`` action — Confirmed sits alongside Direct/
Authored/deferred as a fourth tier value (ADR-0024: a human confirms an
irreversible operation, not curator-drafted content, so it is not Authored).
This module still writes nothing: the delete + reindex live in ``pages.py`` /
``routes.py``, mirroring how ``qa.refile`` owns C9's write path.

Slice S7 scope (Lint Remediation tier-B — issue #383, ADR-0027)
-----------------------------------------------------------------
No new check. C1/C2 flip from Authored to a fourth-and-a-half tier, Routed
(ADR-0027): the fill for a coverage gap or a red link routes through the
existing Upload -> Import -> Ingest pipeline, so no draft ever exists for a
curator to approve — Authored's defining gate would gate nothing.
``_REMEDIATION_TAXONOMY["C1"]`` and ``["C2"]`` move to
``RemediationDescriptor("routed", route="import")`` — ``RemediationDescriptor``
gains a ``route`` field (``None`` for every other tier) so the ONE shared
taxonomy still drives all three surfaces: Console turns the disabled tier-B
placeholder into a real "Fill via Import" navigation control (no execution,
no gate — it commits nothing itself), CLI/MCP render the route as plain text
("fill via: kb import ..."). This module still runs no new check and writes
nothing; the fill itself is entirely the existing Import/Ingest machinery.

Read-only invariant
-------------------
``run_lint()`` does NOT modify wiki page frontmatter.  It writes only:
  - ``wiki/lint-report.md``   (Generated index artifact, gitignored)
  - ``wiki/log.md``           (Runtime trace, append-only)

Concurrency
-----------
``run_lint()`` holds ``indexer._index_lock`` for the full duration so a
concurrent ``/ingest`` (which also holds the lock) cannot produce mid-write
state for lint to observe.  ``/chat`` reads are not blocked.

Continue-on-error
-----------------
If a check raises, the exception is caught, a ``lint_check_error`` log entry is
written, and the error is recorded in ``LintResponse.check_errors``.  Other
checks still run.  The report is always written.

Check execution order (cheapest to most expensive)
---------------------------------------------------
1. C11 orphan (read frontmatter only)
2. C3 failed-grounding (read frontmatter only)
3. C4-a slug collision (filename list only)
4. C6 stale pages (stat docs/ files)
5. C2 red links (scan wiki page bodies)
6. C1 coverage gap (read log.md only)
7. C8 promotion candidates (read qa frontmatter)
8. C9 qa-staleness (stat entity files vs qa updated)
9. C10 qa-schema-validity (read qa frontmatter)
10. C5 page-pair LLM (F1∪F3 filter + LLM) — most expensive last

Authorised by PRD #65 (Phase 5), GitHub issue #66 (Slice 5-1), GitHub issue #67 (Slice 5-2), GitHub issue #68 (Slice 5-3), GitHub issue #69 (Slice 5-4), GitHub issue #70 (Slice 5-5), PRD #78 (Phase 6), GitHub issue #82 (Slice 6-5 Phase 5 amendment), ADR-0023 (Lint Remediation Direct vs Authored), PRD #359 (Lint Remediation tier-A), GitHub issue #361 (Slice S1 — Lint Axis taxonomy), GitHub issue #363 (Slice S3 — Console axis grouping + per-row Direct Remediation + auto-relint), and GitHub issue #365 (Slice S5 — Console zh/en language toggle).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from ._paths import DOCS_DIR, WIKI_DIR
from .atomic import write_text_atomic
from .indexer import _index_lock
from .logger import LOG_PATH, log_event
from .schemas import (
    CollisionDifferentiateDraft,
    CollisionMergeDraft,
    CoverageGapFinding,
    FailedGroundingFinding,
    InvalidQaSchemaFinding,
    LintFindings,
    LintResponse,
    LintSummary,
    OrphanPageFinding,
    PagePairFinding,
    PromotionCandidateFinding,
    QaStalenessFinding,
    ReconcileDraft,
    RedLinkFinding,
    SlugCollisionFinding,
    StalePageFinding,
)

# ---------------------------------------------------------------------------
# C5 — Lazy LLM singleton (ADR-0005 pattern)
# ---------------------------------------------------------------------------

# Module-level sentinel; monkeypatched in tests.
_lint_llm = None

# Best-effort C5 metrics for cost accounting / report honesty. Mutable lists so
# _check_c5_page_pair can write them and run_lint can read-then-reset, without a
# global statement (consistent with existing module-level patterns). Index 0
# holds the current value; run_lint reads and resets both after C5 runs.
#   _c5_llm_call_counter — pairs actually sent to the LLM judge (== judged count, ≤ cap)
#   _c5_capped_counter   — candidate pairs NOT judged because they fell below the cap
#   _c5_pair_errors      — per-pair error strings accumulated during this run; run_lint
#                          reads and resets, then writes to check_errors["c5"] so the
#                          continue-on-error per-pair failures surface in the SUCCESS
#                          payload (server.py docstring / ADR-0016 contract).
# Counting happens once in _check_c5_page_pair (not per-call inside
# _judge_page_pair) so the bounded-concurrency judge has no shared-counter race.
_c5_llm_call_counter: list[int] = [0]
_c5_capped_counter: list[int] = [0]
_c5_pair_errors: list[str] = []


def get_lint_llm():
    """Return the lazy singleton ChatOpenAI for C5 page-pair contradiction detection.

    Model resolution (three-layer fallback, mirroring OPENAI_INGEST_MODEL):
        OPENAI_LINT_MODEL  →  OPENAI_MODEL  →  "gpt-4o-mini"

    temperature=0 for determinism (structured output, reproducible runs).
    """
    global _lint_llm
    if _lint_llm is None:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv(
            "OPENAI_LINT_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        _lint_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
    return _lint_llm


# Regex that matches a trailing ``-N`` suffix where N is an integer >= 2.
# Used by C4-a to strip the ingest-appended collision suffix from a slug.
_COLLISION_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")

# ---------------------------------------------------------------------------
# Module-level default paths (monkeypatched in tests)
# ---------------------------------------------------------------------------

# Re-exported so tests can monkeypatch ``app.lint.WIKI_DIR`` and ``app.lint.DOCS_DIR``
# using the same setattr pattern as the parent conftest's ``_redirect_paths_to_tmp``.
# These must be module attributes (not local variables) for monkeypatch to work.

# WIKI_DIR and DOCS_DIR are imported from ._paths above.
# LOG_PATH is imported from .logger above.


# ---------------------------------------------------------------------------
# C11 — Orphan page detection
# ---------------------------------------------------------------------------


def _orphan_predicate(sources: list[str], docs_filenames: set[str]) -> tuple[bool, list[str]]:
    """Recompute the C11 orphan predicate for one page's frontmatter ``sources``.

    Returns ``(full, missing_deduped)``:
    - ``missing_deduped`` — the missing citation basenames, de-duplicated in
      first-seen order (the existing C11 rendering convention).
    - ``full`` — True iff ``sources`` carries at least one citation with a
      non-empty file part AND every such citation's file is missing under
      ``docs_filenames`` (ADR-0025's full-orphan predicate: "``sources``
      non-empty and every citation's file missing under ``docs/**``"). A
      page whose ``sources`` entries are all blank never counts as full —
      nothing has been confirmed gone.

    Shared by ``_check_c11_orphan`` (the bulk sweep) and ``check_full_orphan``
    (the ``DELETE /pages/{slug}`` re-verification entry point) so the two can
    never disagree about what counts as a full orphan.
    """
    missing: list[str] = []
    valid_citations = 0
    for citation in sources:
        # citation format: "filename.md#anchor"  or just "filename.md"
        file_part = citation.split("#")[0].strip()
        if not file_part:
            continue
        valid_citations += 1
        basename = Path(file_part).name
        if basename not in docs_filenames:
            missing.append(basename)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            deduped.append(m)

    full = valid_citations > 0 and len(missing) == valid_citations
    return full, deduped


def check_full_orphan(sources: list[str], docs_dir: Path) -> bool:
    """Recompute the ADR-0025 full-orphan predicate for one page at delete time.

    Public re-verification entry point for ``pages.delete_full_orphan`` /
    ``DELETE /pages/{slug}`` (ADR-0025 Invariant: "recomputes the full-orphan
    predicate server-side at delete time... never trusts the client's lint
    finding"). Shares ``_orphan_predicate`` with the bulk C11 sweep, so a
    Source restored or re-imported since a lint report rendered is always
    reflected here, and the two call sites can never disagree.
    """
    docs_filenames: set[str] = {p.name for p in docs_dir.glob("**/*.md")}
    full, _missing = _orphan_predicate(sources, docs_filenames)
    return full


def _check_c11_orphan(
    wiki_dir: Path,
    docs_dir: Path,
) -> list[OrphanPageFinding]:
    """Return orphan findings for every wiki page with a missing source file.

    A wiki page is an *orphan* when at least one entry in ``frontmatter.sources``
    references a file (the portion before ``#``) that does not exist anywhere
    under ``docs_dir`` (including nested subdirectories, per ``glob("**/*.md")``).

    The check reads ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.

    Algorithm:
    1. Collect the set of all source filenames (stems with extension) present
       under ``docs_dir`` using ``glob("**/*.md")``.
    2. For each wiki page in ``entities/`` and ``concepts/``, parse its YAML
       frontmatter and read the ``sources`` list.
    3. Recompute the missing-citations + full/partial predicate via
       ``_orphan_predicate`` (tier-B S5, issue #381, ADR-0025) — the same
       helper ``check_full_orphan`` re-runs at delete time.
    4. If any sources are missing, emit one ``OrphanPageFinding`` per page,
       carrying the full/partial distinction (CONTEXT.md "Orphan Page").
    5. Return findings sorted alphabetically by ``page_slug``.

    Only the basename of each source file is matched against the docs glob
    results.  This matches the pattern used by ``/ingest`` (which references
    sources as bare filenames, e.g. ``refund_policy.md#cancellation-window``).
    """
    # Build set of all source file basenames under docs_dir
    docs_filenames: set[str] = {p.name for p in docs_dir.glob("**/*.md")}

    findings: list[OrphanPageFinding] = []

    # Scan both wiki subdirs
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            slug = page_path.stem
            sources = _read_frontmatter_sources(page_path)
            if not sources:
                continue
            full, deduped = _orphan_predicate(sources, docs_filenames)
            if deduped:
                findings.append(
                    OrphanPageFinding(
                        page_slug=slug,
                        missing_sources=deduped,
                        suggested_action=(
                            f"The source(s) {deduped!r} referenced by '{slug}' no longer "
                            f"exist under docs/. If the Source was renamed, update this "
                            f"page's frontmatter sources field and re-ingest. If the Source "
                            f"was deleted, delete this wiki page as it has no valid source."
                        ),
                        full=full,
                    )
                )

    findings.sort(key=lambda f: f.page_slug)
    return findings


def _read_frontmatter_sources(page_path: Path) -> list[str]:
    """Parse the YAML frontmatter of a wiki page and return the ``sources`` list.

    Returns an empty list if the page has no frontmatter, the frontmatter
    cannot be parsed, or the ``sources`` field is absent/empty.

    Delegates to ``_parse_frontmatter`` (fence-*line* scanning) rather than
    requiring the file to *start* with ``---``: real ``/ingest``-produced
    entities/concepts pages open with a sentinel HTML comment before the
    fence, so a ``startswith("---")`` reader returned ``[]`` for every real
    page and C11 could never fire on the actual corpus — the same byte-shape
    bug that once made C8/C9/C10 skip every real Filed Answer (see
    ``_parse_frontmatter``'s docstring). Found live when tier-B S5 made C11
    executable (issue #381).
    """
    fm = _parse_frontmatter(page_path)
    if fm is None:
        return []

    sources = fm.get("sources", [])
    if not isinstance(sources, list):
        return []
    return [str(s) for s in sources if s]


def _parse_frontmatter(page_path: Path) -> dict | None:
    """Parse the YAML frontmatter of a wiki page; return the dict or None.

    Locates the first ``---`` … ``---`` fence by scanning for fence *lines* (a
    line that is exactly ``---`` after stripping) rather than requiring the file
    to *start* with ``---``. Filed qa pages are written with a leading sentinel
    HTML comment before the fence (``<!-- Auto-filed by POST /chat… -->``), so a
    ``startswith("---")`` test skipped every filed draft — which made C8/C9/C10
    silently ignore real Filed Answers and mis-flag them as invalid schema (#4).
    This mirrors ``qa._read_frontmatter``, which already parses these files.

    Returns None if the page has no fence pair, the block cannot be parsed, or
    the result is not a dict.
    """
    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = text.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        return None

    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None

    if not isinstance(fm, dict):
        return None
    return fm


def _iter_wiki_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under entities/ and concepts/."""
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            yield page_path.stem, page_path


# ---------------------------------------------------------------------------
# C3 — Failed-grounding sweep
# ---------------------------------------------------------------------------


def _check_c3_failed_grounding(
    wiki_dir: Path,
) -> list[FailedGroundingFinding]:
    """Return findings for every wiki page with ``frontmatter.status == "failed_grounding"``.

    Phase 3 fail-soft ingest writes these pages when the grounding verifier
    rejects claims or is unavailable.  Phase 4 W1 silently excludes them from
    ``/chat`` retrieval.  C3 surfaces them so the curator can decide whether to
    review the Source and re-ingest or simply delete the page.

    Algorithm:
    1. Iterate ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.
    2. For each page, parse YAML frontmatter.
    3. Skip pages whose ``status`` is not ``"failed_grounding"``.
    4. Build a ``FailedGroundingFinding`` from ``sources[0]``, the nested
       ``grounding_failure`` block (``reason`` + ``unsupported_claims``), and a
       suggested action.
    5. Return findings sorted alphabetically by ``page_slug``.

    If ``grounding_failure`` is absent or malformed, the finding still records
    ``reason="verifier_unavailable"`` and an empty ``unsupported_claims`` list
    rather than raising — defensive because older ingest code may not have
    written the block consistently.
    """
    findings: list[FailedGroundingFinding] = []

    for slug, page_path in _iter_wiki_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "failed_grounding":
            continue

        sources = fm.get("sources", [])
        source_ref = str(sources[0]) if sources else ""

        # Extract grounding_failure sub-block defensively
        gf_raw = fm.get("grounding_failure")
        if isinstance(gf_raw, dict):
            reason = gf_raw.get("reason", "verifier_unavailable")
            if reason not in ("claim_unsupported", "verifier_unavailable"):
                reason = "verifier_unavailable"
            raw_claims = gf_raw.get("unsupported_claims", [])
            unsupported_claims = (
                [str(c) for c in raw_claims] if isinstance(raw_claims, list) else []
            )
        else:
            reason = "verifier_unavailable"
            unsupported_claims = []

        suggested_action = (
            f"Page '{slug}' failed grounding check (reason: {reason}). "
            f"Review the source '{source_ref}' to confirm the claims are supported, "
            f"then re-ingest the source to regenerate this page. "
            f"If the source no longer covers these claims, delete this page."
        )

        findings.append(
            FailedGroundingFinding(
                page_slug=slug,
                source=source_ref,
                reason=reason,  # type: ignore[arg-type]
                unsupported_claims=unsupported_claims,
                suggested_action=suggested_action,
            )
        )

    findings.sort(key=lambda f: f.page_slug)
    return findings


# ---------------------------------------------------------------------------
# C4-a — Slug collision groups
# ---------------------------------------------------------------------------

# Sentinel stamped onto every group member by a successful
# POST /pages/collision/differentiate/apply (issue #378, ADR-0028). Lives here
# (not reconcile.py, which writes it) so the writer template and the C4-a
# exemption parser below cannot drift apart; reconcile.py imports it.
DIFFERENTIATE_SENTINEL_TEMPLATE = (
    "<!-- Differentiated by POST /pages/collision/differentiate/apply on {ts}\n"
    "     (collision group: '{group}').\n"
    "     Grounded in the union of every group member's Sources, re-verified at apply time.\n"
    "     Manual edits are safe until the next reconcile/collision resolution or ingest of the\n"
    "     underlying Source(s) — edit the Source for a permanent change. -->"
)

_DIFFERENTIATE_SENTINEL_RE = re.compile(
    r"<!-- Differentiated by POST /pages/collision/differentiate/apply on [^\n]*\n"
    r"\s*\(collision group: '([^']*)'\)\."
)


def _differentiated_group(page_path: Path) -> set[str] | None:
    """The collision-group member set recorded by a differentiate sentinel.

    Reads the head of ``page_path`` and parses the sentinel that
    ``reconcile._write_differentiated_page`` stamps on every member of a
    differentiated group. Returns ``None`` when the page carries no sentinel
    (never differentiated, or rewritten since — ingest replaces the whole
    file, so a re-ingested page correctly loses its exemption).
    """
    try:
        head = page_path.read_text(encoding="utf-8", errors="replace")[:600]
    except OSError:
        return None
    m = _DIFFERENTIATE_SENTINEL_RE.match(head)
    if not m:
        return None
    return {slug.strip() for slug in m.group(1).split(",") if slug.strip()}


def _check_c4a_slug_collision(
    wiki_dir: Path,
) -> list[SlugCollisionFinding]:
    """Return collision groups for slugs sharing a common base (stripped ``-N`` suffix).

    Phase 3 ingest appends ``-2``, ``-3``, ... suffixes to avoid overwriting
    existing pages.  These collisions indicate that two pages cover the same
    concept (or nearly so) and a curator should review them for merge or
    heading rename.

    Only suffixes with N >= 2 trigger grouping (``-1`` is not an ingest-appended
    collision suffix).

    Algorithm:
    1. Collect all page slugs from ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.
    2. For each slug, test ``_COLLISION_SUFFIX_RE`` (matches ``<base>-<N>`` where
       N is a numeric string).  If N >= 2, the slug belongs to the group for
       ``<base>``; otherwise the slug is itself a base slug.
    3. A group must contain at least 2 members (the base slug + at least one
       suffixed variant, or two suffixed variants with a shared base).
    4. Emit one ``SlugCollisionFinding`` per qualifying group.
    5. Sort: group size descending; alphabetical by ``base_slug`` for ties.

    Cross-directory collisions are included (a slug in ``entities/`` and a
    suffixed variant in ``concepts/`` are grouped together) since the ingest
    uniqueness guarantee is wiki-wide, not per-subdirectory.

    **Differentiate exemption** (issue #378 AC "apply → re-lint clears the
    finding"): a successful differentiate apply stamps every member with a
    sentinel recording the exact group it resolved. A group is skipped when
    every current member carries a sentinel for exactly this member set — the
    curator has ruled the pages intentionally complementary, so the naming
    pattern is no longer evidence of an unresolved collision. A new member
    joining (set mismatch) or an ingest rewrite (sentinel gone) re-fires it.
    """
    # Map base_slug → set of member slugs
    groups: dict[str, set[str]] = {}
    paths: dict[str, Path] = {}

    for slug, page_path in _iter_wiki_pages(wiki_dir):
        paths[slug] = page_path
        m = _COLLISION_SUFFIX_RE.match(slug)
        if m and int(m.group(2)) >= 2:
            # Suffixed variant (`pricing-2`, `pricing-3`, ...): group under its base.
            # Do NOT seed the group with {base} — the unsuffixed base page may not
            # exist on disk; report only slugs that actually exist.
            groups.setdefault(m.group(1), set()).add(slug)
        else:
            # No suffix (or -1, which is not a collision-appended variant): the slug
            # is its own base. Always add so iteration order does not affect which
            # slugs land in a pre-existing group keyed by this same base.
            groups.setdefault(slug, set()).add(slug)

    findings: list[SlugCollisionFinding] = []
    for base_slug, members in groups.items():
        if len(members) < 2:
            continue
        if all(_differentiated_group(paths[slug]) == members for slug in members):
            continue
        pages_in_group = sorted(members)
        suggested_action = (
            f"Slug collision: {len(members)} pages share the base slug '{base_slug}' "
            f"({', '.join(pages_in_group)}). "
            f"Review the pages and either merge them into a single page or rename their "
            f"headings to be more specific so ingest assigns distinct slugs."
        )
        findings.append(
            SlugCollisionFinding(
                base_slug=base_slug,
                pages_in_group=pages_in_group,
                suggested_action=suggested_action,
            )
        )

    # Sort: group size descending, then alphabetical by base_slug for ties
    findings.sort(key=lambda f: (-len(f.pages_in_group), f.base_slug))
    return findings


def find_inbound_references(slug: str, wiki_dir: Path) -> tuple[list[str], list[str]]:
    """Return ``(wiki_referrer_slugs, qa_referrer_slugs)`` for ``slug``.

    Used by the C4 merge-apply reference guard (``reconcile.py``, ADR-0028
    Invariant) — a variant with ANY inbound reference refuses deletion. Two
    distinct reference mechanisms, mirroring the checks that already scan
    for each:

    - **Wiki referrers** — ``entities/``/``concepts/`` pages with a resolved
      ``[[slug]]`` wikilink pointing at ``slug`` (same regex C2 uses for red
      links, but here the target DOES exist — this is the inverse case).
    - **Qa referrers** — ``wiki/qa/*.md`` Filed Answers whose
      ``frontmatter.sources`` cites ``slug`` (bare or ``slug#heading``),
      regardless of ``status`` — a draft citing a soon-deleted page is still
      a real reference (C9's citation-extraction convention, but unfiltered
      by status: the guard errs toward refusing).

    Both lists are sorted for deterministic output; ``slug`` itself is never
    included (a page cannot reference itself as an inbound link for this
    purpose).
    """
    wiki_referrers: set[str] = set()
    for page_slug, page_path in _iter_wiki_pages(wiki_dir):
        if page_slug == slug:
            continue
        try:
            body = page_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in _WIKILINK_RE.finditer(body):
            if match.group(1).strip() == slug:
                wiki_referrers.add(page_slug)
                break

    qa_referrers: set[str] = set()
    for qa_slug, qa_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(qa_path)
        if fm is None:
            continue
        for citation in fm.get("sources", []) or []:
            if str(citation).split("#", 1)[0].strip() == slug:
                qa_referrers.add(qa_slug)
                break

    return sorted(wiki_referrers), sorted(qa_referrers)


# ---------------------------------------------------------------------------
# C6 — mtime-based stale detection
# ---------------------------------------------------------------------------


def _check_c6_stale(
    wiki_dir: Path,
    docs_dir: Path,
) -> list[StalePageFinding]:
    """Return stale findings for every wiki page whose Source content has changed.

    A wiki page is *stale* when:
    - ``frontmatter.source_hashes`` contains an entry for the Source file, AND
    - The stored ``docs_body`` SHA-256 hash differs from the current file content's hash.

    Pages with no ``source_hashes`` frontmatter (legacy Phase 6 pages, drift state unknown)
    are skipped rather than generating false positives. Pages whose Source file does NOT
    exist are handled by C11 (orphan check). C6 explicitly skips them.

    Hash comparison is stable across ``git clone``/checkout — unlike mtime, the
    SHA-256 of file content does not change when the working tree is reconstructed from
    the same commit. ``source_mtime`` and ``page_updated`` are still read for display
    purposes in the emitted finding.

    Algorithm:
    1. For each wiki page in ``entities/`` and ``concepts/``:
       a. Parse frontmatter; skip if no sources or no source_hashes.
       b. Take ``sources[0]``; strip ``#anchor`` to get the Source filename.
       c. Look up ``source_hashes[<filename>]["docs_body"]``; skip if absent/None.
       d. Resolve ``docs_dir / <filename>``; if the file does not exist, skip (C11's job).
       e. Compute ``SHA-256(source_path.read_text("utf-8").encode())``.
       f. If current hash != stored hash, emit ``StalePageFinding``.
    2. Return findings sorted by ``drift_days`` descending.
    """
    findings: list[StalePageFinding] = []

    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            slug = page_path.stem
            fm = _parse_frontmatter(page_path)
            if fm is None:
                continue

            sources = fm.get("sources", [])
            if not isinstance(sources, list) or not sources:
                continue

            # C6 uses only the first source citation
            first_citation = str(sources[0]) if sources[0] else ""
            if not first_citation:
                continue

            # Strip anchor to get the Source filename
            file_part = first_citation.split("#")[0].strip()
            if not file_part:
                continue
            source_filename = Path(file_part).name

            # Check for stored content hash.  Missing or empty source_hashes means
            # drift state is unknown (legacy page never ingested with hash tracking):
            # skip to avoid false positives.
            source_hashes = fm.get("source_hashes")
            if not source_hashes or not isinstance(source_hashes, dict):
                continue
            hash_entry = source_hashes.get(source_filename)
            if not isinstance(hash_entry, dict):
                continue
            stored_hash = hash_entry.get("docs_body")
            if stored_hash is None:
                continue

            # Resolve Source file; skip if missing (C11's job)
            source_path = docs_dir / source_filename
            if not source_path.exists():
                # Also try nested lookup using glob
                matches = list(docs_dir.glob(f"**/{source_filename}"))
                if not matches:
                    continue
                source_path = matches[0]

            # Compute current content hash (same algorithm as ingest._compute_docs_body_hash)
            current_hash = hashlib.sha256(
                source_path.read_text(encoding="utf-8").encode()
            ).hexdigest()

            if current_hash == stored_hash:
                # Content unchanged — not stale (git clone / mtime drift is invisible here)
                continue

            # Content has changed: emit finding.  source_mtime and page_updated are
            # informational — the detection gate is the hash, not the mtime.
            source_mtime_ts = source_path.stat().st_mtime
            source_mtime = datetime.datetime.fromtimestamp(source_mtime_ts, tz=datetime.UTC)

            # Parse page's updated timestamp
            updated_str = fm.get("updated", "")
            if not updated_str:
                continue
            try:
                page_updated = datetime.datetime.fromisoformat(
                    str(updated_str).replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if page_updated.tzinfo is None:
                page_updated = page_updated.replace(tzinfo=datetime.UTC)

            drift_seconds = max(0.0, (source_mtime - page_updated).total_seconds())
            drift_days = drift_seconds / 86400.0
            findings.append(
                StalePageFinding(
                    page_slug=slug,
                    source=source_filename,
                    source_mtime=source_mtime,
                    page_updated=page_updated,
                    drift_days=drift_days,
                    suggested_action=(
                        f"Source '{source_filename}' content has changed since wiki page "
                        f"'{slug}' was last ingested. Re-ingest the Source to synchronise "
                        f'the wiki page: POST /ingest {{"source": "{source_filename}"}}.'
                    ),
                )
            )

    # Sort by drift_days descending
    findings.sort(key=lambda f: f.drift_days, reverse=True)
    return findings


# ---------------------------------------------------------------------------
# C2 — Red link backlog
# ---------------------------------------------------------------------------

# Regex from the AC: captures slug portions of [[slug]] and [[slug#anchor]] and [[slug|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

# Files in wiki root that must NEVER contribute red links (self-feeding loop + noise)
_C2_EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "lint-report.md",
        "index.md",
        "log.md",
        "hot.md",
        "README.md",
    }
)


def _check_c2_red_links(
    wiki_dir: Path,
) -> list[RedLinkFinding]:
    """Return red link findings for every unresolved ``[[wikilink]]`` slug.

    Scans ``wiki/entities/`` and ``wiki/concepts/`` ONLY (matching ADR-0006 SOURCE_DIRS).
    ``wiki/.archive/*`` and root-level special files are explicitly excluded.

    Algorithm:
    1. Build the set of existing page slugs from ``entities/*.md`` + ``concepts/*.md``.
    2. For each page in those dirs, scan the page body for ``[[...]]`` patterns.
       Skip files in ``_C2_EXCLUDED_FILENAMES`` (by basename).
       Skip files in ``wiki/.archive/``.
    3. For each wikilink, extract the slug portion (drop ``#anchor`` and ``|alias``).
       If the slug is NOT in the existing slugs set, it is a red link.
    4. Aggregate by slug: count total occurrences; track pages that reference it;
       capture ~50-char context from the first occurrence.
    5. Return findings sorted by ``mention_count`` descending, alphabetical by ``slug`` for ties.

    Heading anchors (``[[slug#heading]]``) are captured but only the slug is checked.
    """
    # Build the set of known page slugs
    existing_slugs: set[str] = set()
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            existing_slugs.add(page_path.stem)

    # Per-slug aggregation: mention_count, referenced_by set, first context
    slug_counts: dict[str, int] = defaultdict(int)
    slug_pages: dict[str, set[str]] = defaultdict(set)
    slug_first_context: dict[str, str | None] = {}

    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            # Exclusion: skip files by name
            if page_path.name in _C2_EXCLUDED_FILENAMES:
                continue
            # Exclusion: skip .archive/ files
            if ".archive" in page_path.parts:
                continue

            page_slug = page_path.stem
            try:
                body = page_path.read_text(encoding="utf-8")
            except OSError:
                continue

            # Find all wikilinks in the body
            for match in _WIKILINK_RE.finditer(body):
                target_slug = match.group(1).strip()
                if not target_slug:
                    continue
                if target_slug in existing_slugs:
                    # Resolved — not a red link
                    continue
                # Unresolved red link
                slug_counts[target_slug] += 1
                slug_pages[target_slug].add(page_slug)
                # Capture context from first occurrence only
                if target_slug not in slug_first_context:
                    start = match.start()
                    # Take ~25 chars before and ~25 chars after the match
                    ctx_start = max(0, start - 25)
                    ctx_end = min(len(body), match.end() + 25)
                    slug_first_context[target_slug] = body[ctx_start:ctx_end]

    findings: list[RedLinkFinding] = []
    for slug, count in slug_counts.items():
        findings.append(
            RedLinkFinding(
                slug=slug,
                mention_count=count,
                referenced_by=sorted(slug_pages[slug]),
                sample_context=slug_first_context.get(slug),
            )
        )

    # Sort by mention_count descending, then alphabetical by slug for ties
    findings.sort(key=lambda f: (-f.mention_count, f.slug))
    return findings


# ---------------------------------------------------------------------------
# C1 — Coverage gap aggregation from chat_fallback / chat_grounding_fallback log
# ---------------------------------------------------------------------------

# Regex to match a log line produced by logger.log_event:
#   ## [<ISO-8601>] <kind> | <summary>
_LOG_LINE_RE = re.compile(r"^## \[(?P<ts>[^\]]+)\] (?P<kind>[^ ]+) \| (?P<summary>.*)$")

# Regex to extract the leading double-quoted query from a summary field.
# Matches: "<query>" ... (query may contain single-quotes; the retrieval
# module replaces double-quotes inside queries with single-quotes before logging)
_SUMMARY_QUERY_RE = re.compile(r'^"(?P<query>[^"]*)"')

# Log kinds consumed by C1
_C1_KINDS = frozenset({"chat_fallback", "chat_grounding_fallback"})

# Reason values that C1 handles explicitly
_C1_HANDLED_REASONS = frozenset({"retrieval_empty", "below_threshold", "claim_unsupported"})


def _canonicalise(q: str) -> str:
    """Return the canonical cluster key for a query string.

    Rules (per issue #69 AC):
    - Lowercase
    - Strip leading and trailing punctuation characters
    - Collapse internal whitespace to single spaces
    - Strip outer whitespace
    - No stop-word removal
    - No token sorting
    """
    # Lowercase
    result = q.lower()
    # Strip outer whitespace
    result = result.strip()
    # Strip leading/trailing punctuation (all chars in string.punctuation)
    result = result.strip(string.punctuation)
    # Strip any remaining outer whitespace after punctuation stripping
    result = result.strip()
    # Collapse internal whitespace to single spaces
    result = re.sub(r"\s+", " ", result)
    return result


def _parse_kv(summary: str) -> dict[str, str]:
    """Parse key=value pairs from a log summary string.

    Supports bare values (no quotes) and ignores the leading quoted query
    string.  Returns a dict of all key=value pairs found.
    """
    # Remove the leading quoted query (may contain spaces) first
    remainder = _SUMMARY_QUERY_RE.sub("", summary).strip()
    pairs: dict[str, str] = {}
    for match in re.finditer(r"(\w+)=(\S+)", remainder):
        pairs[match.group(1)] = match.group(2)
    return pairs


def _check_c1_coverage_gaps(log_path: Path) -> list[CoverageGapFinding]:
    """Parse ``wiki/log.md`` and aggregate coverage gap findings.

    Reads the log file line by line. Matches ``chat_fallback`` and
    ``chat_grounding_fallback`` entries. Groups them into clusters according
    to the per-reason cluster key table in issue #69:

    - ``retrieval_empty``: key = ``_canonicalise(q)``
    - ``below_threshold``: key = ``(_canonicalise(q), top_section)``
    - ``claim_unsupported``: key = ``(_canonicalise(q), tuple(sorted(cited_pages)))``
    - ``wiki_layer_empty``: skipped entirely (filtered by kind, not reason)

    Env var ``KB_LINT_MIN_HITS`` (default 1): clusters with ``hit_count <
    KB_LINT_MIN_HITS`` are excluded from the returned list.  Read at function
    entry (not at module load) to mirror the ``KB_SCORE_THRESHOLD`` pattern.

    Counter semantics:
    - ``malformed_lines``: lines that are in a C1 kind but whose structure
      could not be parsed (missing query field, regex non-match, parse exception).
      If ``malformed_lines > 0`` a ``lint_check_error`` is written.
    - ``out_of_scope_reasons``: lines parsed cleanly but whose ``reason=`` value
      is not one C1 handles (e.g. ``not_indexed``, ``verifier_unavailable``).
      These are silently ignored — they are legitimately not C1's concern.

    Returns findings sorted:
    - Primary: fixed group order (``retrieval_empty``, ``below_threshold``,
      ``claim_unsupported``)
    - Within group: ``hit_count`` descending; ties broken alphabetically by
      ``query_canonical``
    """
    min_hits = int(os.getenv("KB_LINT_MIN_HITS", "1"))

    if not log_path.exists():
        return []

    # Cluster accumulators keyed by (reason, cluster_key)
    # Each value: {"hits": int, "raw_queries_seen": [str], "timestamps": [str],
    #              "top_section": str|None, "cited_pages": list[str]|None}
    clusters: dict[tuple[str, Any], dict[str, Any]] = {}

    malformed_lines = 0
    out_of_scope_reasons = 0

    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _LOG_LINE_RE.match(line)
        if not m:
            # Not a structured log line (e.g., blank header, arbitrary text); skip
            continue

        ts = m.group("ts")
        kind = m.group("kind")
        summary = m.group("summary")

        if kind not in _C1_KINDS:
            continue

        # Extract leading quoted query
        qm = _SUMMARY_QUERY_RE.match(summary)
        if not qm:
            malformed_lines += 1
            continue
        raw_query = qm.group("query")

        # Parse key=value pairs from remainder
        try:
            kv = _parse_kv(summary)
        except Exception:  # noqa: BLE001
            malformed_lines += 1
            continue

        reason = kv.get("reason", "")

        if reason not in _C1_HANDLED_REASONS:
            # Parsed cleanly but not a reason C1 handles — not a parse failure
            out_of_scope_reasons += 1
            continue

        canonical = _canonicalise(raw_query)

        if reason == "retrieval_empty":
            cluster_key: Any = canonical
            top_section: str | None = None
            cited_pages: list[str] | None = None

        elif reason == "below_threshold":
            top_section = kv.get("top_section")
            cluster_key = (canonical, top_section)
            cited_pages = None

        else:  # claim_unsupported
            cited_raw = kv.get("cited", "")
            cited_list: list[str] = [c for c in cited_raw.split(",") if c] if cited_raw else []
            cited_pages = cited_list if cited_list else None
            cluster_key = (canonical, tuple(sorted(cited_list)))
            top_section = None

        full_key = (reason, cluster_key)

        if full_key not in clusters:
            clusters[full_key] = {
                "reason": reason,
                "query_canonical": canonical,
                "hits": 0,
                "raw_queries_seen": [],  # ordered, deduplicated up to 3
                "raw_seen_set": set(),
                "timestamps": [],
                "top_section": top_section,
                "cited_pages": cited_pages,
            }

        entry = clusters[full_key]
        entry["hits"] += 1
        entry["timestamps"].append(ts)

        # Collect up to 3 unique raw queries
        if len(entry["raw_queries_seen"]) < 3 and raw_query not in entry["raw_seen_set"]:
            entry["raw_queries_seen"].append(raw_query)
            entry["raw_seen_set"].add(raw_query)

    if malformed_lines > 0:
        log_event(
            "lint_check_error",
            f"check=c1 malformed_lines={malformed_lines} out_of_scope_reasons={out_of_scope_reasons} reason=malformed_log_lines",
            log_path=log_path,
        )

    # Build CoverageGapFinding list, applying min_hits filter
    findings: list[CoverageGapFinding] = []
    for (_reason, _ck), entry in clusters.items():
        if entry["hits"] < min_hits:
            continue
        reason = entry["reason"]
        canonical = entry["query_canonical"]
        raw_q = entry["raw_queries_seen"]
        timestamps = sorted(entry["timestamps"])
        first_seen = timestamps[0] if timestamps else ""
        last_seen = timestamps[-1] if timestamps else ""

        ts_ = entry["top_section"]
        cp = entry["cited_pages"]

        if reason == "retrieval_empty":
            action = f"Create a new wiki page covering {canonical}"
        elif reason == "below_threshold":
            section_ref = ts_ or "<unknown>"
            action = f"Extend page `[[{section_ref}]]` to cover {canonical}"
        else:  # claim_unsupported
            pages_str = ", ".join(cp) if cp else "<unknown>"
            action = f"Review: KB gap or verifier issue? cited: {pages_str}"

        findings.append(
            CoverageGapFinding(
                reason=reason,
                query_canonical=canonical,
                sample_raw_queries=raw_q,
                hit_count=entry["hits"],
                first_seen=first_seen,
                last_seen=last_seen,
                top_section=ts_,
                cited_pages=cp,
                suggested_action=action,
            )
        )

    # Sort: fixed group order, then hit_count desc, then query_canonical asc
    _GROUP_ORDER = {"retrieval_empty": 0, "below_threshold": 1, "claim_unsupported": 2}
    findings.sort(key=lambda f: (_GROUP_ORDER.get(f.reason, 99), -f.hit_count, f.query_canonical))

    return findings


# ---------------------------------------------------------------------------
# C5 — Page-pair LLM contradiction detection
# ---------------------------------------------------------------------------

# env vars for C5 BM25 candidate filter
_KB_LINT_BM25_TOP_K_DEFAULT = 3
_KB_LINT_BM25_THRESHOLD_DEFAULT = 1.0

# env vars for the C5 scaling fix (issue #194)
# KB_LINT_C5_MAX_PAIRS: judge at most this many candidate pairs (top-K by
#   similarity). Caps LLM cost/time to a constant regardless of corpus size.
# KB_LINT_C5_CONCURRENCY: bounded worker count for the surviving LLM calls.
_KB_LINT_C5_MAX_PAIRS_DEFAULT = 30
_KB_LINT_C5_CONCURRENCY_DEFAULT = 5


def _load_wiki_pages(wiki_dir: Path) -> dict[str, dict]:
    """Load all wiki pages from entities/, concepts/, and qa/ into a dict.

    Returns a dict mapping slug → {
        "slug": str,
        "body": str (full file text after frontmatter),
        "sources": list[str],
        "path": Path,
        "type": str | None ("entity" | "concept" | "qa", from frontmatter),
    }

    Used by _candidate_pairs and _check_c5_page_pair to avoid re-parsing
    frontmatter multiple times. The ``type`` entry is what the Slice 6-5 C5
    modifier reads to exclude ``type == "qa"`` pages from candidate pair
    generation (PRD #78 Phase 5 amendment).

    qa pages are included here so the C5 modifier filter sees them and drops
    them; downstream consumers (e.g. ``_candidate_pairs``) own the filter. The
    set returned is the entire wiki page corpus (entities/concepts/qa), not the
    filtered subset.
    """
    pages: dict[str, dict] = {}
    for slug, page_path in _iter_all_wiki_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        sources: list[str] = []
        page_type: str | None = None
        if fm:
            raw_sources = fm.get("sources", [])
            if isinstance(raw_sources, list):
                sources = [str(s) for s in raw_sources if s]
            raw_type = fm.get("type")
            if isinstance(raw_type, str):
                page_type = raw_type
        try:
            full_text = page_path.read_text(encoding="utf-8")
        except OSError:
            full_text = ""
        # Strip frontmatter to get body
        body = full_text
        if full_text.startswith("---"):
            parts = full_text.split("---", 2)
            body = parts[2] if len(parts) >= 3 else full_text
        pages[slug] = {
            "slug": slug,
            "body": body.strip(),
            "sources": sources,
            "path": page_path,
            "type": page_type,
        }
    return pages


def _iter_all_wiki_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under entities/, concepts/, qa/.

    Distinct from ``_iter_wiki_pages`` (entities + concepts only) — used by
    ``_load_wiki_pages`` so the C5 modifier (Slice 6-5) sees qa pages and can
    filter them out at the pair-generation layer.
    """
    # Phase 6 Slice 6-5: include qa/ so the modifier filter can see and drop them.
    for subdir_name in ("entities", "concepts", "qa"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            yield page_path.stem, page_path


def _candidate_pairs(
    pages: dict[str, dict],
    wiki_dir: Path,
) -> set[tuple[str, str]]:
    """Return the F1 ∪ F3 candidate pair set for C5 LLM judgement.

    F1: pairs of pages whose ``frontmatter.sources`` lists share at least one
        source citation.

    F3: for each page, treat its body as a BM25 query and call
        ``indexer.search(body, k=KB_LINT_BM25_TOP_K)``.  Any returned Section
        whose score exceeds ``KB_LINT_BM25_THRESHOLD`` and whose page slug is
        different from the query page contributes a candidate pair.

    All pair tuples are canonicalised: ``(min(a, b), max(a, b))`` so that
    ``(A, B)`` and ``(B, A)`` deduplicate to a single pair.  This is the
    symmetric pair short-circuit invariant — each pair is judged exactly once.

    ``KB_LINT_BM25_TOP_K`` env var (default 3) controls per-page BM25 hits.
    ``KB_LINT_BM25_THRESHOLD`` env var (default 1.0) controls the score gate.
    Both are read at call time (not module load) to mirror KB_SCORE_THRESHOLD.

    Phase 6 Slice 6-5 (C5 modifier — PRD #78)
    -----------------------------------------
    Pages whose ``frontmatter.type == "qa"`` are excluded from candidate pair
    generation BEFORE F1/F3 are computed. Filed Answers are structurally
    derivative of their source entity pages (they share the same
    ``frontmatter.sources``), so without this filter the C5 LLM would be called
    on every (qa, entity-source) pair only to return ``severity=duplicate``,
    flooding ``lint-report.md`` and burning LLM tokens. Filtering at the page
    set level means the qa pages neither appear in F1 source-intersection nor
    as F3 BM25 queries, *and* the F3 BM25 hit-side filter rejects sections
    whose owner is a qa page — qa is fully invisible to C5 by construction.
    """
    # Read env vars at call time
    bm25_top_k = int(os.getenv("KB_LINT_BM25_TOP_K", str(_KB_LINT_BM25_TOP_K_DEFAULT)))
    bm25_threshold = float(
        os.getenv("KB_LINT_BM25_THRESHOLD", str(_KB_LINT_BM25_THRESHOLD_DEFAULT))
    )

    # C5 modifier (Slice 6-5): drop type=qa pages from the candidate pool BEFORE
    # F1/F3 candidate computation so the LLM call budget excludes qa pairs entirely.
    non_qa_pages: dict[str, dict] = {
        slug: data for slug, data in pages.items() if data.get("type") != "qa"
    }

    pairs: set[tuple[str, str]] = set()
    slug_list = list(non_qa_pages.keys())

    # F1: shared sources (only among non-qa pages)
    for i, slug_a in enumerate(slug_list):
        srcs_a = set(non_qa_pages[slug_a]["sources"])
        if not srcs_a:
            continue
        for slug_b in slug_list[i + 1 :]:
            srcs_b = set(non_qa_pages[slug_b]["sources"])
            if srcs_a & srcs_b:
                pair = (min(slug_a, slug_b), max(slug_a, slug_b))
                pairs.add(pair)

    # F3: BM25 self-query
    # Build a temporary index from wiki pages so we can BM25-query body text.
    # We reuse indexer.parse_markdown via a tmp directory approach, but since the
    # index is already built from wiki pages (by /index), we use indexer.search
    # directly which queries the in-memory sections list.
    #
    # NOTE: The in-memory index must be populated before calling run_lint().
    # This is the normal case (bot needs an index to function). If the index is
    # empty (sections list empty), F3 produces no pairs — safe degradation.
    from .indexer import search as bm25_search

    for slug_a, data_a in non_qa_pages.items():
        body_a = data_a["body"]
        if not body_a.strip():
            continue
        try:
            hits = bm25_search(body_a, k=bm25_top_k)
        except Exception:  # noqa: BLE001
            continue
        for section, score in hits:
            if score < bm25_threshold:
                continue
            # section.file is the bare slug (ADR-0006: wiki pages indexed with slug as source_id)
            slug_b = section.file
            if slug_b == slug_a:
                continue
            # Only add pairs where both slugs exist in the non-qa page set —
            # this is the hit-side half of the C5 qa filter: even if the index
            # surfaces a qa-page section, it cannot enter the candidate pool.
            if slug_b not in non_qa_pages:
                continue
            pair = (min(slug_a, slug_b), max(slug_a, slug_b))
            pairs.add(pair)

    return pairs


# ---------------------------------------------------------------------------
# C5 similarity pre-filter (issue #194) — rank candidate pairs by lexical
# token-overlap so the LLM judge only sees the top-K most-similar pairs.
# ---------------------------------------------------------------------------


def _body_tokens(body: str) -> frozenset[str]:
    """Tokenise a page body into a set of comparison tokens.

    Reuses ``indexer.tokenize`` — the same tokeniser BM25 retrieval (and the F3
    candidate filter) already use — so the similarity signal is consistent with
    the rest of the lint pipeline and inherits its CJK-bigram + stop-word
    handling (Phase 16) for free. A set (not a multiset) is sufficient: Jaccard
    is defined over sets and the goal is only a cheap relative ranking.
    """
    from .indexer import tokenize

    return frozenset(tokenize(body))


def _pair_similarity(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    """Return the Jaccard similarity |A∩B| / |A∪B| of two token sets (0.0–1.0).

    "Shares a source" (F1) is a weak proxy for "might contradict"; topical token
    overlap is the better discriminator for which pairs are worth an LLM call.
    Empty-token pages score 0.0 (they sort to the bottom and are capped first).
    """
    if not tokens_a or not tokens_b:
        return 0.0
    union = len(tokens_a | tokens_b)
    return len(tokens_a & tokens_b) / union if union else 0.0


def _rank_candidate_pairs(
    pairs: set[tuple[str, str]],
    pages: dict[str, dict],
) -> list[tuple[str, str]]:
    """Return ``pairs`` ordered most- to least-similar (deterministic).

    Sort key: similarity descending, then the canonical ``(page_a, page_b)``
    tuple ascending as a stable tie-break so the ranking — and therefore which
    pairs survive the top-K cap — is fully reproducible across runs.
    """
    token_cache: dict[str, frozenset[str]] = {}
    for slug_a, slug_b in pairs:
        if slug_a not in token_cache:
            token_cache[slug_a] = _body_tokens(pages[slug_a]["body"])
        if slug_b not in token_cache:
            token_cache[slug_b] = _body_tokens(pages[slug_b]["body"])

    def sort_key(pair: tuple[str, str]) -> tuple[float, str, str]:
        a, b = pair
        return (-_pair_similarity(token_cache[a], token_cache[b]), a, b)

    return sorted(pairs, key=sort_key)


# Prompt for the C5 LLM call — instructs the model to compare two wiki page bodies
_C5_SYSTEM_PROMPT = """You are a knowledge-base health auditor. Two wiki pages from the same knowledge base are shown below. Your task is to judge whether they contradict, overlap, or duplicate each other.

Output a structured finding with:
- severity: one of "direct" (explicit factual conflict — different numbers, different policies), "tension" (same topic, scope or wording differences that could confuse readers), "duplicate" (same concept covered in two pages without contradiction), or "none" (no meaningful overlap or conflict found).
- page_a_claim: a direct quote from Page A's body that is relevant to the comparison. Use the exact text.
- page_b_claim: a direct quote from Page B's body that is relevant to the comparison. Use the exact text.
- summary: a one-to-two sentence explanation of why you assigned this severity.
- suggested_action: a concrete curator action (e.g. "Reconcile sources", "Merge pages", "Review and dismiss").

If severity is "none", still provide page_a_claim and page_b_claim — pick any representative sentence from each page.

Be conservative: only assign "direct" for clear factual disagreement (different numbers, dates, policy terms). Use "tension" for ambiguous cases."""


def _judge_page_pair(
    slug_a: str,
    body_a: str,
    slug_b: str,
    body_b: str,
) -> PagePairFinding:
    """Call the LLM to judge whether two wiki pages contradict, overlap, or duplicate.

    Uses ``get_lint_llm().with_structured_output(PagePairFinding)`` per ADR-0005.
    temperature=0 (set on the singleton in ``get_lint_llm()``).

    Returns a ``PagePairFinding`` with canonical slug ordering enforced:
    ``page_a`` is always the lexicographically-smaller slug.

    LangChain types are confined to this function — callers see only
    ``PagePairFinding`` (ADR-0005 § Consequences).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Ensure canonical slug order
    if slug_a > slug_b:
        slug_a, slug_b = slug_b, slug_a
        body_a, body_b = body_b, body_a

    llm = get_lint_llm()
    chain = llm.with_structured_output(PagePairFinding)

    messages = [
        SystemMessage(content=_C5_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"**Page A** (slug: `{slug_a}`):\n\n{body_a}\n\n"
                f"---\n\n**Page B** (slug: `{slug_b}`):\n\n{body_b}"
            )
        ),
    ]

    finding: PagePairFinding = chain.invoke(messages)

    # Enforce canonical slug order in the finding (LLM may swap them)
    if finding.page_a != slug_a or finding.page_b != slug_b:
        finding = PagePairFinding(
            severity=finding.severity,
            page_a=slug_a,
            page_b=slug_b,
            page_a_claim=finding.page_a_claim,
            page_b_claim=finding.page_b_claim,
            summary=finding.summary,
            suggested_action=finding.suggested_action,
        )

    return finding


# ---------------------------------------------------------------------------
# Reconcile drafting (tier-B S1 — issue #376, ADR-0028)
# ---------------------------------------------------------------------------

_RECONCILE_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Two wiki pages currently disagree about a \
fact (a Coherence contradiction). You are given both pages' current content \
and the union of the Source excerpts they cite. Rewrite BOTH pages so they \
state mutually consistent facts, each one grounded ONLY in the provided \
Source excerpts — never invent a fact absent from them.

Rules:
- Keep each page's own topic; resolve the disagreement, do not merge the \
two pages into one or make either page about the other's topic.
- Preserve the page's existing "# <Heading>" line and its trailing \
"[Source: ...]" citation line verbatim; rewrite only the prose between them.
- Write in the same language as the original page.
- If the Source excerpts do not settle which side of the disagreement is \
correct, state the uncertainty explicitly on both pages rather than \
picking a side arbitrarily.

Return content_a (the full revised content for Page A, structurally \
identical in shape to the original) and content_b (same for Page B).
"""


def _build_reconcile_user_message(
    page_a: str,
    content_a: str,
    page_b: str,
    content_b: str,
    union_sections: list,
) -> str:
    """Format the two pages' current content plus the union Sources for the drafting call."""
    parts = [
        f"**Page A** (slug: `{page_a}`):\n\n{content_a}",
        f"**Page B** (slug: `{page_b}`):\n\n{content_b}",
    ]
    source_parts = []
    for section in union_sections:
        heading = " > ".join(section.heading_path)
        source_parts.append(f"[Source: {section.id}]\nHeading: {heading}\n{section.content}")
    sources_text = "\n\n".join(source_parts) if source_parts else "(no Source excerpts available)"
    parts.append(f"**Cited Source excerpts (union of both pages' Sources):**\n\n{sources_text}")
    return "\n\n---\n\n".join(parts)


def generate_reconcile_draft(
    page_a: str,
    content_a: str,
    page_b: str,
    content_b: str,
    union_sections: list,
) -> ReconcileDraft:
    """Call the LLM to draft mutually-consistent content for two contradicting pages.

    ADR-0028 tracer bullet (C5 Reconcile, ``POST /pages/reconcile``). Uses
    ``get_lint_llm().with_structured_output(ReconcileDraft)`` — the SAME lazy
    singleton ``_judge_page_pair`` uses for C5 judging — so drafting stays
    inside this already-blessed LLM-facing module (ADR-0005) rather than
    opening a second one for a single call site.

    LangChain types are confined to this function (CODING_STANDARD §2.4);
    callers receive a plain ``ReconcileDraft`` Pydantic model.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_lint_llm()
    chain = llm.with_structured_output(ReconcileDraft)

    messages = [
        SystemMessage(content=_RECONCILE_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_reconcile_user_message(
                page_a, content_a, page_b, content_b, union_sections
            )
        ),
    ]

    draft: ReconcileDraft = chain.invoke(messages)
    return draft


# ---------------------------------------------------------------------------
# Collision drafting (tier-B S2 — issue #378, ADR-0028)
# ---------------------------------------------------------------------------

_COLLISION_MERGE_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Several wiki pages were auto-suffixed \
because they cover the same concept (a Coherence slug collision). You are \
given the current content of every page in the group and the union of the \
Source excerpts they cite. Draft ONE merged page that covers everything the \
group's pages state, grounded ONLY in the provided Source excerpts — never \
invent a fact absent from them.

Rules:
- Preserve the base page's existing "# <Heading>" line and its trailing \
"[Source: ...]" citation line verbatim; rewrite only the prose between them.
- Fold in every distinct fact from the other pages in the group; do not \
just repeat the base page unchanged.
- Write in the same language as the original pages.
- Resolve near-duplicate phrasing into one clear statement; do not simply \
concatenate the pages.

Return content_base: the full merged content for the base page, structurally \
identical in shape to the original base page.
"""


def _build_collision_merge_user_message(
    base_slug: str,
    base_content: str,
    variant_contents: dict[str, str],
    union_sections: list,
) -> str:
    """Format the base page + every variant's current content plus the union
    Sources for the merge drafting call."""
    parts = [f"**Base page** (slug: `{base_slug}`):\n\n{base_content}"]
    for variant_slug, variant_content in variant_contents.items():
        parts.append(f"**Variant** (slug: `{variant_slug}`):\n\n{variant_content}")
    source_parts = []
    for section in union_sections:
        heading = " > ".join(section.heading_path)
        source_parts.append(f"[Source: {section.id}]\nHeading: {heading}\n{section.content}")
    sources_text = "\n\n".join(source_parts) if source_parts else "(no Source excerpts available)"
    parts.append(
        f"**Cited Source excerpts (union of every group member's Sources):**\n\n{sources_text}"
    )
    return "\n\n---\n\n".join(parts)


def generate_collision_merge_draft(
    base_slug: str,
    base_content: str,
    variant_contents: dict[str, str],
    union_sections: list,
) -> CollisionMergeDraft:
    """Call the LLM to draft one merged page for a C4 collision group.

    ADR-0028 tier-B S2 (``POST /pages/collision/merge``). Uses
    ``get_lint_llm().with_structured_output(CollisionMergeDraft)`` — the SAME
    lazy singleton every other lint LLM call site uses.

    LangChain types are confined to this function (CODING_STANDARD §2.4);
    callers receive a plain ``CollisionMergeDraft`` Pydantic model.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_lint_llm()
    chain = llm.with_structured_output(CollisionMergeDraft)

    messages = [
        SystemMessage(content=_COLLISION_MERGE_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_collision_merge_user_message(
                base_slug, base_content, variant_contents, union_sections
            )
        ),
    ]

    draft: CollisionMergeDraft = chain.invoke(messages)
    return draft


_COLLISION_DIFFERENTIATE_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Several wiki pages were auto-suffixed \
because they cover the same concept (a Coherence slug collision). You are \
given the current content of every page in the group and the union of the \
Source excerpts they cite. Rewrite EVERY page so the group becomes \
complementary and more specific — each page keeps its own distinct angle on \
the concept and none of them are dropped — grounded ONLY in the provided \
Source excerpts — never invent a fact absent from them.

Rules:
- Preserve each page's existing "# <Heading>" line and its trailing \
"[Source: ...]" citation line verbatim; rewrite only the prose between them.
- Do not merge the pages into one; every slug in the group must keep its \
own page with distinct, non-duplicated content.
- Write in the same language as the original pages.

Return pages: one entry per input slug, each with its full revised content, \
structurally identical in shape to the original.
"""


def _build_collision_differentiate_user_message(
    contents: dict[str, str],
    union_sections: list,
) -> str:
    """Format every group member's current content plus the union Sources
    for the differentiate drafting call."""
    parts = [f"**Page** (slug: `{slug}`):\n\n{content}" for slug, content in contents.items()]
    source_parts = []
    for section in union_sections:
        heading = " > ".join(section.heading_path)
        source_parts.append(f"[Source: {section.id}]\nHeading: {heading}\n{section.content}")
    sources_text = "\n\n".join(source_parts) if source_parts else "(no Source excerpts available)"
    parts.append(
        f"**Cited Source excerpts (union of every group member's Sources):**\n\n{sources_text}"
    )
    return "\n\n---\n\n".join(parts)


def generate_collision_differentiate_draft(
    contents: dict[str, str],
    union_sections: list,
) -> CollisionDifferentiateDraft:
    """Call the LLM to draft complementary content for every page in a C4
    collision group.

    ADR-0028 tier-B S2 (``POST /pages/collision/differentiate``). Uses
    ``get_lint_llm().with_structured_output(CollisionDifferentiateDraft)`` —
    the SAME lazy singleton every other lint LLM call site uses.

    LangChain types are confined to this function (CODING_STANDARD §2.4);
    callers receive a plain ``CollisionDifferentiateDraft`` Pydantic model.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_lint_llm()
    chain = llm.with_structured_output(CollisionDifferentiateDraft)

    messages = [
        SystemMessage(content=_COLLISION_DIFFERENTIATE_SYSTEM_PROMPT),
        HumanMessage(content=_build_collision_differentiate_user_message(contents, union_sections)),
    ]

    draft: CollisionDifferentiateDraft = chain.invoke(messages)
    return draft


def _check_c5_page_pair(
    wiki_dir: Path,
) -> list[PagePairFinding]:
    """Run C5 page-pair contradiction detection over the wiki.

    Steps:
    1. Load all wiki pages (slug, body, sources) via ``_load_wiki_pages``.
    2. Build candidate pairs via ``_candidate_pairs`` (F1 ∪ F3 filter).
    3. Rank candidates by lexical similarity and judge at most
       ``KB_LINT_C5_MAX_PAIRS`` (env, default 30) of them — the similarity
       pre-filter (issue #194). Pairs below the cap are NOT judged; their count
       is recorded in ``_c5_capped_counter`` so the report can surface them as
       "not judged (capped)" rather than dropping them silently.
    4. Judge the surviving pairs via ``_judge_page_pair``.
    5. Filter out findings with severity == "none".
    6. Continue-on-error: if the LLM raises for a pair, log the pair skipped
       and retain prior findings.

    Records C5 run metrics for ``run_lint`` (read-then-reset there):
    ``_c5_llm_call_counter`` (judged == LLM calls) and ``_c5_capped_counter``
    (candidates not judged because they fell below the cap).

    Returns findings sorted by severity order (direct → tension → duplicate),
    then alphabetically by page_a slug. The sort makes output order independent
    of judge completion order, so concurrency does not affect the result.
    """
    max_pairs = int(os.getenv("KB_LINT_C5_MAX_PAIRS", str(_KB_LINT_C5_MAX_PAIRS_DEFAULT)))
    if max_pairs < 0:
        max_pairs = _KB_LINT_C5_MAX_PAIRS_DEFAULT
    concurrency = int(os.getenv("KB_LINT_C5_CONCURRENCY", str(_KB_LINT_C5_CONCURRENCY_DEFAULT)))
    if concurrency < 1:
        concurrency = _KB_LINT_C5_CONCURRENCY_DEFAULT

    pages = _load_wiki_pages(wiki_dir)
    pairs = _candidate_pairs(pages, wiki_dir)

    ranked = _rank_candidate_pairs(pairs, pages)
    # Drop any pair whose pages vanished (defensive — candidates come from pages).
    judged_pairs = [(a, b) for a, b in ranked[:max_pairs] if a in pages and b in pages]
    capped_pairs = ranked[max_pairs:]

    # Record metrics up front so they reflect intent even if judging is empty.
    _c5_capped_counter[0] = len(capped_pairs)
    _c5_llm_call_counter[0] = len(judged_pairs)

    findings: list[PagePairFinding] = []
    errors: list[str] = []

    # Reset the module-level per-pair error accumulator for this run.
    _c5_pair_errors.clear()

    def _judge(pair: tuple[str, str]) -> PagePairFinding:
        slug_a, slug_b = pair
        return _judge_page_pair(slug_a, pages[slug_a]["body"], slug_b, pages[slug_b]["body"])

    # Bounded concurrency: the surviving (≤cap) calls are network-bound LLM
    # round-trips, so a small thread pool cuts wall-time without unbounded fan-out.
    # Continue-on-error is per-pair: one failed judge never sinks the others.
    if judged_pairs:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(judged_pairs))) as executor:
            future_to_pair = {executor.submit(_judge, pair): pair for pair in judged_pairs}
            for future in as_completed(future_to_pair):
                slug_a, slug_b = future_to_pair[future]
                try:
                    finding = future.result()
                    if finding.severity != "none":
                        findings.append(finding)
                except Exception as exc:  # noqa: BLE001
                    err = f"({slug_a},{slug_b}): {type(exc).__name__}: {exc}"
                    errors.append(err)
                    _c5_pair_errors.append(err)

    if errors:
        # Log errors without breaking; the check returns partial results.
        # Per-pair error strings are also written to _c5_pair_errors so run_lint
        # can surface them in check_errors["c5"] (SUCCESS payload, not isError).
        log_event(
            "lint_check_error",
            f"check=c5 pairs_failed={len(errors)} first_error={errors[0][:200]}",
        )

    # Sort: severity order, then alphabetical by page_a
    _SEVERITY_ORDER = {"direct": 0, "tension": 1, "duplicate": 2, "none": 3}
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), f.page_a, f.page_b))

    return findings


# ---------------------------------------------------------------------------
# C8 / C9 / C10 — Phase 5 amendment for Phase 6 Filed Answers (PRD #78)
# ---------------------------------------------------------------------------

# Valid frontmatter.status values for Filed Answer pages (Slice 6-1 schema).
# Used by C10 to validate ``status`` field; ``status`` outside this set is a
# curator-typo orphan-zombie risk (Q8d in PRD #78).
_VALID_QA_STATUS_VALUES: frozenset[str] = frozenset({"live", "draft", "stale", "superseded"})


def _iter_qa_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under wiki/qa/.

    Separate iterator from ``_iter_wiki_pages`` because the qa lifecycle is
    distinct (Filed Answer lifecycle owned by PRD #78, not the entity/concept
    ingest lifecycle) and the three Phase 5 amendment checks (C8/C9/C10) scan
    only qa/, never entities/ or concepts/.
    """
    subdir = wiki_dir / "qa"
    if not subdir.exists():
        return
    for page_path in sorted(subdir.glob("*.md")):
        yield page_path.stem, page_path


# Default cap on Promotion Candidates surfaced per /lint run.
_KB_LINT_PROMOTION_TOP_N_DEFAULT = 10

# Truncation length for Promotion Candidate question field (keeps the report
# table column width manageable; readers can open the page for the full text).
_PROMOTION_CANDIDATE_QUESTION_MAXLEN = 80


def _check_c8_promotion_candidates(
    wiki_dir: Path,
) -> list[PromotionCandidateFinding]:
    """Surface ``status: draft`` Filed Answers ranked by re-ask popularity.

    PRD #78 Phase 5 amendment §"C8 — promotion candidates". Read-only — never
    mutates frontmatter; the actual draft→live promotion is Phase 6's
    ``POST /qa/{slug}/promote`` (Slice 6-4).

    Algorithm:
    1. Scan ``wiki/qa/*.md``.
    2. Skip pages whose frontmatter cannot be parsed (defensive — C10 surfaces
       schema breakage independently).
    3. Skip pages whose ``status`` is not ``"draft"``.
    4. Build PromotionCandidateFinding with ``slug``, truncated ``question``,
       ``count``, ``age_days`` (now - created in UTC), and ``cited_count``.
    5. Sort by ``count`` desc, then ``updated`` desc (tiebreak).
    6. Cap at the top ``KB_LINT_PROMOTION_TOP_N`` (env var, default 10).

    Returns the ranked, capped list. Empty list when no qa pages exist or no
    drafts are present.
    """
    top_n = int(os.getenv("KB_LINT_PROMOTION_TOP_N", str(_KB_LINT_PROMOTION_TOP_N_DEFAULT)))

    # Each candidate carries a sort tuple so we can rank, then strip back to the
    # Finding shape for the public return.
    candidates: list[tuple[int, str, PromotionCandidateFinding]] = []

    now_utc = datetime.datetime.now(datetime.UTC)

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "draft":
            continue

        # Question text — truncate for the report column. Falls back to the
        # slug if question is missing (defensive; C10 separately surfaces a
        # missing_question finding).
        raw_question = fm.get("question") or ""
        question = str(raw_question).strip()
        if not question:
            question = f"(missing question — slug `{slug}`)"
        if len(question) > _PROMOTION_CANDIDATE_QUESTION_MAXLEN:
            question = question[: _PROMOTION_CANDIDATE_QUESTION_MAXLEN - 1] + "…"

        # Count — coerce defensively; missing/invalid -> 1 (the schema default
        # value, also documented in WikiPageFrontmatter).
        raw_count = fm.get("count", 1)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 1

        # Cited count — number of citation entries in frontmatter.sources.
        raw_sources = fm.get("sources", [])
        cited_count = len(raw_sources) if isinstance(raw_sources, list) else 0

        # Age in days — now() - frontmatter.created. Missing/unparseable falls
        # back to 0.0 (rendered as 0.0 days; non-fatal because C10 surfaces the
        # underlying schema issue if any field is wrong).
        age_days = 0.0
        created_str = fm.get("created", "")
        if created_str:
            try:
                created_dt = datetime.datetime.fromisoformat(
                    str(created_str).replace("Z", "+00:00")
                )
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=datetime.UTC)
                age_days = (now_utc - created_dt).total_seconds() / 86400.0
            except ValueError:
                age_days = 0.0

        # Sort tiebreaker — frontmatter.updated as ISO string. Lexicographic
        # comparison is correct for ISO-8601 UTC strings; missing -> empty
        # string which sorts last.
        updated_key = str(fm.get("updated", ""))

        finding = PromotionCandidateFinding(
            slug=slug,
            question=question,
            count=count,
            age_days=round(age_days, 1),
            cited_count=cited_count,
        )
        candidates.append((count, updated_key, finding))

    # Rank with two stable passes (Python's sort is stable, so secondary keys
    # propagate cleanly): first sort ascending by slug (final tiebreaker),
    # then ascending by updated (so a desc-updated stable sort by count desc
    # produces the AC-specified order: count desc, then updated desc).
    # Single-pass tuple sort would require negating the string updated key,
    # which is brittle for unequal-length ISO timestamps; two-pass is clearer.
    candidates.sort(key=lambda t: t[2].slug)  # tertiary: slug asc
    candidates.sort(key=lambda t: t[1], reverse=True)  # secondary: updated desc
    candidates.sort(key=lambda t: t[0], reverse=True)  # primary: count desc

    return [finding for _c, _u, finding in candidates[:top_n]]


def _check_c9_qa_staleness(
    wiki_dir: Path,
) -> list[QaStalenessFinding]:
    """Flag live Filed Answers whose cited entity pages have been re-ingested more recently.

    PRD #78 Phase 5 amendment §"C9 — qa-staleness". Read-only — surfaced to
    ``lint-report.md`` §``## Stale Filed Answers``. Closes Q6b "entity
    re-ingested, qa stranded" failure mode.

    Staleness is detected by comparing the entity page's ``frontmatter.updated``
    timestamp against the qa page's ``frontmatter.updated`` timestamp.  This is
    stable across ``git clone``/checkout because both timestamps come from file
    *content*, not filesystem metadata.  When an entity is re-ingested, ``/ingest``
    writes a new ``updated`` value into the entity frontmatter, which advances the
    timestamp and triggers this check.

    Algorithm:
    1. For each ``wiki/qa/*.md`` with ``frontmatter.status == "live"``:
       a. Parse ``frontmatter.sources``; for each citation
          ``"<entity-slug>#<heading-slug>"`` extract the bare entity slug.
       b. For each entity slug, locate the entity file: try
          ``wiki/entities/<slug>.md`` first, then ``wiki/concepts/<slug>.md``.
       c. Parse the entity page's ``frontmatter.updated`` as a UTC datetime.
          If entity_updated > qa.frontmatter.updated, the citation is "stale".
       d. If at least one citation is stale, emit a QaStalenessFinding with
          all stale citations and the max drift in days.
    2. Return findings sorted by ``max_drift_days`` desc, then slug asc.

    Pages whose ``frontmatter.updated`` cannot be parsed are skipped (defensive;
    C10 surfaces schema breakage separately).
    """
    findings: list[QaStalenessFinding] = []

    qa_dir = wiki_dir / "qa"
    if not qa_dir.exists():
        return findings

    # Build a lookup map: entity-slug -> Path (entities/ first, concepts/ second).
    # Filed Answer citations point at wiki entity/concept slugs (per ADR-0006
    # §Citation contract — wiki Section citations look like "<page-slug>#<heading>").
    entity_lookup: dict[str, Path] = {}
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            entity_slug = page_path.stem
            # entities/ wins over concepts/ on collision (deterministic).
            entity_lookup.setdefault(entity_slug, page_path)

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "live":
            continue

        updated_str = fm.get("updated", "")
        if not updated_str:
            continue
        try:
            qa_updated = datetime.datetime.fromisoformat(str(updated_str).replace("Z", "+00:00"))
        except ValueError:
            continue
        if qa_updated.tzinfo is None:
            qa_updated = qa_updated.replace(tzinfo=datetime.UTC)

        raw_sources = fm.get("sources", [])
        if not isinstance(raw_sources, list) or not raw_sources:
            continue

        stale_citations: list[str] = []
        max_drift_seconds = 0.0
        for citation in raw_sources:
            if not citation:
                continue
            citation_str = str(citation)
            # Citation shape: "<entity-slug>#<heading-slug>" OR bare "<slug>"
            entity_slug = citation_str.split("#", 1)[0].strip()
            if not entity_slug:
                continue
            entity_path = entity_lookup.get(entity_slug)
            if entity_path is None or not entity_path.exists():
                # Missing entity file — not C9's concern; C9 only flags re-ingest
                # drift. (No C-check currently flags qa→missing-entity; that's a
                # potential future C9.b or new check.)
                continue

            # Parse the entity page's frontmatter.updated for content-stable comparison.
            entity_fm = _parse_frontmatter(entity_path)
            if entity_fm is None:
                continue
            entity_updated_str = entity_fm.get("updated", "")
            if not entity_updated_str:
                continue
            try:
                entity_updated = datetime.datetime.fromisoformat(
                    str(entity_updated_str).replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if entity_updated.tzinfo is None:
                entity_updated = entity_updated.replace(tzinfo=datetime.UTC)

            if entity_updated > qa_updated:
                stale_citations.append(citation_str)
                drift_seconds = (entity_updated - qa_updated).total_seconds()
                if drift_seconds > max_drift_seconds:
                    max_drift_seconds = drift_seconds

        if stale_citations:
            findings.append(
                QaStalenessFinding(
                    page_slug=slug,
                    stale_citations=stale_citations,
                    max_drift_days=round(max_drift_seconds / 86400.0, 1),
                )
            )

    findings.sort(key=lambda f: (-f.max_drift_days, f.page_slug))
    return findings


def _check_c10_qa_schema_validity(
    wiki_dir: Path,
) -> list[InvalidQaSchemaFinding]:
    """Surface qa pages with schema invalidities (status, type, question, count).

    PRD #78 Phase 5 amendment §"C10 — qa-schema-validity". Layer 3 of the
    orphan-visibility defence (PRD #78 Q8d): the indexer logs invalid status,
    the filing layer refuses to touch invalid pages, and C10 surfaces them in
    ``lint-report.md`` where curators habitually look.

    Validation rules (one finding per (page, broken property) tuple):
    - ``status``    ∈ ``{live, draft, stale, superseded}``
    - ``type``      == ``"qa"`` (mismatched page type under wiki/qa/)
    - ``question``  is present and non-empty string
    - ``count``     is present and a positive integer

    Pages whose frontmatter cannot be parsed (no ``---`` fence or YAML error)
    produce a single finding flagging ``frontmatter`` as the broken property —
    the curator sees the page surfaces and can investigate manually.

    Returns findings sorted by ``page_slug``, then ``property_name``.
    """
    findings: list[InvalidQaSchemaFinding] = []

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="frontmatter",
                    offending_value="<unparseable>",
                )
            )
            continue

        # status
        if "status" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="status",
                    offending_value="<missing>",
                )
            )
        else:
            status_val = fm.get("status")
            if not isinstance(status_val, str) or status_val not in _VALID_QA_STATUS_VALUES:
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="status",
                        offending_value=str(status_val),
                    )
                )

        # type
        if "type" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="type",
                    offending_value="<missing>",
                )
            )
        else:
            type_val = fm.get("type")
            if type_val != "qa":
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="type",
                        offending_value=str(type_val),
                    )
                )

        # question — required + non-empty string
        raw_question = fm.get("question")
        if raw_question is None:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="question",
                    offending_value="<missing>",
                )
            )
        elif not isinstance(raw_question, str) or not raw_question.strip():
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="question",
                    offending_value=repr(raw_question),
                )
            )

        # count — required + positive integer
        if "count" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="count",
                    offending_value="<missing>",
                )
            )
        else:
            raw_count = fm.get("count")
            valid_count = False
            if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count > 0:
                valid_count = True
            if not valid_count:
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="count",
                        offending_value=repr(raw_count),
                    )
                )

    findings.sort(key=lambda f: (f.page_slug, f.property_name))
    return findings


# ---------------------------------------------------------------------------
# Lint Axis taxonomy (issue #361 / ADR-0023 / CONTEXT.md "Lint Axis")
# ---------------------------------------------------------------------------


class LintCheckMeta(NamedTuple):
    """One taxonomy entry: a wired check's code, short label, and axis.

    ``label`` is the CONTEXT.md "Lint Axis" short name (English). ``label_zh``
    is its Traditional-Chinese counterpart, added by issue #365 (tier-A S5)
    for the Operator Console's zh/en chrome toggle — both live on the same
    taxonomy entry rather than a separate per-interface table (issue #365 AC
    "single source, no per-interface duplication"). The written report / CLI
    / MCP renderers stay English-only for now; the Console is the first (and
    so far only) ``label_zh`` consumer.
    """

    code: str
    label: str
    axis: str
    label_zh: str


# Stable axis order per CONTEXT.md "Lint Axis": Freshness -> Coherence ->
# Coverage -> Lifecycle. The report renderer and group_findings_by_axis both
# walk this order so a later slice can reuse it for the CLI/MCP surfaces too
# (ADR-0017 interface parity, per ADR-0023 "one taxonomy, three interfaces").
LINT_AXIS_ORDER: tuple[str, ...] = ("Freshness", "Coherence", "Coverage", "Lifecycle")

# Traditional-Chinese display strings for LINT_AXIS_ORDER's axis identifiers
# (issue #365). The identifiers themselves stay English — they are stable
# keys used throughout the report renderer / CLI / MCP / tests; only the
# rendered heading text is bilingual, and only the Console renders it today.
LINT_AXIS_LABEL_ZH: dict[str, str] = {
    "Freshness": "新鮮度",
    "Coherence": "一致性",
    "Coverage": "覆蓋率",
    "Lifecycle": "生命週期",
}

# code -> LintCheckMeta for all ten wired checks. Entries are grouped by axis
# in LINT_AXIS_ORDER, and ordered within each axis exactly as CONTEXT.md
# enumerates them — group_findings_by_axis relies on this iteration order.
LINT_CHECK_TAXONOMY: dict[str, LintCheckMeta] = {
    "C6": LintCheckMeta("C6", "stale", "Freshness", "過時"),
    "C3": LintCheckMeta("C3", "failed-grounding", "Freshness", "驗證失敗"),
    "C11": LintCheckMeta("C11", "orphan", "Freshness", "孤立頁面"),
    "C5": LintCheckMeta("C5", "contradiction", "Coherence", "矛盾"),
    "C4": LintCheckMeta("C4", "collision", "Coherence", "重複"),
    "C1": LintCheckMeta("C1", "coverage-gap", "Coverage", "覆蓋缺口"),
    "C2": LintCheckMeta("C2", "red-link", "Coverage", "失效連結"),
    "C8": LintCheckMeta("C8", "promotion", "Lifecycle", "待升級"),
    "C10": LintCheckMeta("C10", "invalid-schema", "Lifecycle", "格式錯誤"),
    "C9": LintCheckMeta("C9", "stale-qa", "Lifecycle", "資料過舊"),
}

# code -> LintFindings attribute name, so group_findings_by_axis can pull each
# check's finding list without a check-specific if/elif chain.
_FINDINGS_ATTR_BY_CODE: dict[str, str] = {
    "C11": "orphans",
    "C3": "failed_grounding",
    "C4": "slug_collisions",
    "C6": "stale_pages",
    "C2": "red_links",
    "C1": "coverage_gaps",
    "C5": "page_pairs",
    "C8": "promotion_candidates",
    "C9": "stale_filed_answers",
    "C10": "invalid_qa_schemas",
}


class LintAxisGroup(NamedTuple):
    """One axis's slice of a lint run: the axis name plus its checks, each
    paired with that check's finding list, in CONTEXT.md order."""

    axis: str
    checks: list[tuple[LintCheckMeta, list[Any]]]


def group_findings_by_axis(findings: LintFindings) -> list[LintAxisGroup]:
    """Group a lint run's findings into axis -> check -> findings.

    Returns one ``LintAxisGroup`` per axis, in ``LINT_AXIS_ORDER``
    (Freshness -> Coherence -> Coverage -> Lifecycle); within each axis,
    checks appear in ``LINT_CHECK_TAXONOMY``'s per-axis order. A check with
    zero findings still gets an entry (empty list) — this helper never drops
    a check; the renderer applies the empty-section convention on top.

    Pure data transform: does not run any check, only reshapes an already
    computed ``LintFindings``.
    """
    groups: list[LintAxisGroup] = []
    for axis in LINT_AXIS_ORDER:
        checks: list[tuple[LintCheckMeta, list[Any]]] = [
            (meta, getattr(findings, _FINDINGS_ATTR_BY_CODE[code]))
            for code, meta in LINT_CHECK_TAXONOMY.items()
            if meta.axis == axis
        ]
        groups.append(LintAxisGroup(axis=axis, checks=checks))
    return groups


# ---------------------------------------------------------------------------
# Remediation descriptor (issue #363 / ADR-0023 tier-A S3)
# ---------------------------------------------------------------------------


class RemediationAction(NamedTuple):
    """One executable Remediation operation wired to an *existing* endpoint.

    ``verb`` is the curator-facing action name (``"reingest"``,
    ``"reingest_retry"``, ``"discard"``, ``"promote"``, ``"delete"``).
    ``target_field`` names the finding attribute that supplies the request
    value (e.g. ``StalePageFinding.source``, ``InvalidQaSchemaFinding.
    page_slug``) — a consumer reads this field off the finding to build the
    request body/path; the field name itself is never re-derived client-side
    (CODING_STANDARD §12.5 no business logic in the client).
    ``force`` is ``True`` only for C3's retry re-ingest: without
    ``force=True`` on ``POST /ingest``, hash-skip idempotency (#93) no-ops
    the retry into a false fix (ADR-0023 Invariant).
    """

    verb: str
    target_field: str
    force: bool = False


class RemediationDescriptor(NamedTuple):
    """A check's Remediation tier plus its executable actions.

    ``actions`` is empty for most ``"authored"`` findings, every ``"routed"``
    one, and every ``"deferred"`` one — tier alone drives a Direct-only
    consumer's disabled tier-B affordance (Authored), a navigation control
    (Routed), or "no control yet" rendering (deferred). C9 is the one
    Authored exception (tier-B S4, issue #380, ADR-0026): its ``refile``
    action carries a real one-click-to-open remediation (the human gate is
    the downstream Promote, not a preview step here) — see the C9 entry in
    ``_REMEDIATION_TAXONOMY`` below. C11 is the one Confirmed check with an
    action (tier-B S5, issue #381, ADR-0024/0025): its ``delete`` action
    opens a confirmation naming the operation, not a draft-review —
    Confirmed's human gate is "confirm this happens", never "approve this
    content". ``route`` is set only for Routed checks (C1/C2, tier-B S7,
    issue #383, ADR-0027) — it names the existing workflow the finding's
    fill navigates to (currently always ``"import"``); every other tier
    leaves it ``None``. Routed has no gate and no action to execute: the
    affordance it drives is pure navigation, so a consumer never mistakes it
    for something it could batch or approve.
    """

    tier: str  # "direct" | "authored" | "confirmed" | "routed" | "deferred"
    actions: tuple[RemediationAction, ...] = ()
    route: str | None = None


# code -> RemediationDescriptor. Direct-tier actions wire to the *existing*
# endpoints named in ADR-0023 Consequences (zero new endpoints): C6/C3 ->
# POST /ingest, C10 -> DELETE /qa/{slug}, C8 -> POST /qa/{slug}/promote +
# DELETE /qa/{slug} (rendered by the Curation Queue block, not per-row lint
# buttons — issue #363 AC "C8 promotion controls remain in the dedicated
# Curation Queue block, unchanged"). Authored-tier checks C5/C4 get an empty
# ``actions`` tuple: visible under their axis, never one-click actionable —
# Authored Remediation always has a curator approval gate (ADR-0023) that
# these two checks surface as a preview/edit/apply step (reconcile/collision,
# ADR-0028). C9 is Authored WITH an action (tier-B S4, issue #380, ADR-0026
# decision 1): its own gate is the *downstream* Promote step on the
# resulting draft, not a preview here, so ``refile`` wires directly like a
# Direct action even though the check stays Authored-classified (the
# additive synthesis half is what classifies it — see ADR-0026). C11 orphan
# is ``"confirmed"`` (tier-B S5, issue #381, ADR-0024/0025): a human confirms
# the named irreversible ``DELETE /pages/{slug}`` operation, not curator-
# drafted content, so it is a fourth tier value rather than shoehorned into
# Authored — Confirmed involves no LLM call anywhere and never batches
# (ADR-0024 Invariant). The action wires unconditionally here; the per-finding
# full/partial eligibility (``OrphanPageFinding.full``) is what a consumer
# reads to decide whether to render the delete button or advisory text only.
# C1/C2 are ``"routed"`` (tier-B S7, issue #383, ADR-0027): fill routes
# through the existing Upload -> Import -> Ingest pipeline, so no draft ever
# exists for a curator to approve — Authored's gate would gate nothing.
# Routed carries no action (there is nothing to execute, only to navigate
# to) but DOES carry ``route`` — the one field a Routed descriptor sets — so
# every surface can render the SAME navigation hint off the one shared
# taxonomy (Console: a real "Fill via Import" control; CLI/MCP: the route as
# text). ADR-0027 Invariant: a Routed remediation commits nothing itself.
_REMEDIATION_TAXONOMY: dict[str, RemediationDescriptor] = {
    "C6": RemediationDescriptor("direct", (RemediationAction("reingest", "source"),)),
    "C3": RemediationDescriptor(
        "direct", (RemediationAction("reingest_retry", "source", force=True),)
    ),
    "C11": RemediationDescriptor("confirmed", (RemediationAction("delete", "page_slug"),)),
    "C5": RemediationDescriptor("authored"),
    "C4": RemediationDescriptor("authored"),
    "C1": RemediationDescriptor("routed", route="import"),
    "C2": RemediationDescriptor("routed", route="import"),
    "C8": RemediationDescriptor(
        "direct",
        (
            RemediationAction("promote", "slug"),
            RemediationAction("discard", "slug"),
        ),
    ),
    "C10": RemediationDescriptor("direct", (RemediationAction("discard", "page_slug"),)),
    "C9": RemediationDescriptor("authored", (RemediationAction("refile", "page_slug"),)),
}


def remediation_for(code: str) -> RemediationDescriptor:
    """Return the Remediation tier + actions for a wired check code.

    Pure lookup into ``_REMEDIATION_TAXONOMY`` — the single source of truth
    for which checks are Direct / Authored / Confirmed / Routed / deferred
    (ADR-0023 / ADR-0024 / ADR-0027). ``code`` is one of the ten
    ``LINT_CHECK_TAXONOMY`` keys (a finding *type*, e.g. ``"C6"`` — the
    tier/action/target-field/force shape does not vary per finding
    *instance*, only per check). Raises
    ``KeyError`` for an unknown code so a typo fails loudly rather than
    silently rendering no Remediation.
    """
    return _REMEDIATION_TAXONOMY[code]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _check_heading(code: str, title: str) -> str:
    """Build a check's H3 sub-heading: ``### <CODE> <title> — <axis-label>``.

    ``title`` is each check's existing descriptive heading text (with its
    finding count where applicable, e.g. ``"Failed grounding (2 pages)"``);
    the taxonomy label is appended so every check is clearly identifiable by
    its Lint Axis short name (issue #361 AC), without discarding the more
    descriptive wording readers already know.
    """
    return f"### {code} {title} — {LINT_CHECK_TAXONOMY[code].label}"


def _render_c11_orphans(findings: LintFindings) -> list[str]:
    """C11 Orphan pages — always rendered (empty-section convention)."""
    lines: list[str] = [_check_heading("C11", "Orphan pages"), ""]
    if not findings.orphans:
        lines.append("_No orphan pages found._")
    else:
        for orphan in findings.orphans:
            lines.append(f"#### `{orphan.page_slug}`")
            lines.append("")
            lines.append(
                f"**Missing sources:** {', '.join(f'`{s}`' for s in orphan.missing_sources)}"
            )
            lines.append("")
            # tier-B S5 (issue #381, ADR-0025): full orphans are eligible for the
            # Confirmed delete; partial orphans stay advisory-only (repair, never delete).
            lines.append(f"**Orphan type:** {'full' if orphan.full else 'partial'}")
            lines.append("")
            lines.append(f"**Suggested action:** {orphan.suggested_action}")
            lines.append("")
    lines.append("")
    return lines


def _render_c3_failed_grounding(findings: LintFindings) -> list[str]:
    """C3 Failed grounding — always rendered (empty-section convention)."""
    n_c3 = len(findings.failed_grounding)
    lines: list[str] = [_check_heading("C3", f"Failed grounding ({n_c3} pages)"), ""]
    if not findings.failed_grounding:
        lines.append("_No failed-grounding pages found._")
    else:
        lines.append("| Page slug | Source | Reason | Unsupported claims |")
        lines.append("| --- | --- | --- | --- |")
        for fg in findings.failed_grounding:
            claims_cell = "; ".join(fg.unsupported_claims) if fg.unsupported_claims else "—"
            lines.append(f"| `{fg.page_slug}` | `{fg.source}` | {fg.reason} | {claims_cell} |")
        lines.append("")
        for fg in findings.failed_grounding:
            lines.append(f"**`{fg.page_slug}`** — {fg.suggested_action}")
            lines.append("")
    lines.append("")
    return lines


def _render_c4_slug_collisions(findings: LintFindings) -> list[str]:
    """C4 Slug collision groups — always rendered (empty-section convention)."""
    n_c4 = len(findings.slug_collisions)
    lines: list[str] = [_check_heading("C4", f"Slug collision groups ({n_c4} groups)"), ""]
    if not findings.slug_collisions:
        lines.append("_No slug collision groups found._")
    else:
        lines.append("| Base slug | Pages in group | Group size |")
        lines.append("| --- | --- | --- |")
        for sc in findings.slug_collisions:
            pages_cell = ", ".join(f"`{p}`" for p in sc.pages_in_group)
            lines.append(f"| `{sc.base_slug}` | {pages_cell} | {len(sc.pages_in_group)} |")
        lines.append("")
        for sc in findings.slug_collisions:
            lines.append(f"**`{sc.base_slug}`** — {sc.suggested_action}")
            lines.append("")
    lines.append("")
    return lines


def _render_c6_stale_pages(findings: LintFindings) -> list[str]:
    """C6 Stale pages — always rendered (empty-section convention)."""
    n_stale = len(findings.stale_pages)
    title = f"Stale pages ({n_stale} page{'s' if n_stale != 1 else ''})"
    lines: list[str] = [_check_heading("C6", title), ""]
    if not findings.stale_pages:
        lines.append("_No stale pages found._")
    else:
        lines.append("| Page | Source | Source mtime | Page updated | Drift (days) | Action |")
        lines.append("|------|--------|-------------|--------------|-------------|--------|")
        for stale in findings.stale_pages:
            src_mtime_str = stale.source_mtime.strftime("%Y-%m-%d")
            pg_updated_str = stale.page_updated.strftime("%Y-%m-%d")
            lines.append(
                f"| `{stale.page_slug}` | `{stale.source}` | {src_mtime_str} | {pg_updated_str}"
                f" | {stale.drift_days:.1f} | {stale.suggested_action} |"
            )
    lines.append("")
    return lines


def _render_c2_red_links(findings: LintFindings) -> list[str]:
    """C2 Red links — always rendered (empty-section convention)."""
    n_red = len(findings.red_links)
    title = f"Red links ({n_red} backlog item{'s' if n_red != 1 else ''})"
    lines: list[str] = [_check_heading("C2", title), ""]
    if not findings.red_links:
        lines.append("_No red links found._")
    else:
        lines.append("| Target slug | Mentions | Referenced by | Sample context |")
        lines.append("|-------------|---------|---------------|----------------|")
        for rl in findings.red_links:
            # referenced_by rendered as comma-separated [[slug]] wikilinks
            ref_by_str = ", ".join(f"[[{s}]]" for s in rl.referenced_by)
            ctx_str = rl.sample_context.replace("|", "\\|") if rl.sample_context else ""
            lines.append(f"| `{rl.slug}` | {rl.mention_count} | {ref_by_str} | {ctx_str} |")
    lines.append("")
    return lines


def _render_c1_coverage_gaps(findings: LintFindings) -> list[str]:
    """C1 Coverage gaps — always rendered (empty-section convention)."""
    n_c1 = len(findings.coverage_gaps)
    lines: list[str] = [_check_heading("C1", f"Coverage gaps ({n_c1} findings)"), ""]
    if not findings.coverage_gaps:
        lines.append("_No coverage gaps found._")
        lines.append("")
    else:
        # Sub-group by reason in fixed order
        by_reason: dict[str, list[CoverageGapFinding]] = defaultdict(list)
        for gap in findings.coverage_gaps:
            by_reason[gap.reason].append(gap)

        for reason in ("retrieval_empty", "below_threshold", "claim_unsupported"):
            group = by_reason.get(reason, [])
            if not group:
                continue
            lines.append(f"#### Repeated {reason} ({len(group)})")
            lines.append("")
            for gap in group:
                lines.append(f"- **`{gap.query_canonical}`** (×{gap.hit_count})")
                lines.append(f"  - *{gap.suggested_action}*")
                if gap.sample_raw_queries:
                    samples = "; ".join(f'"{q}"' for q in gap.sample_raw_queries[:3])
                    lines.append(f"  - Sample queries: {samples}")
                lines.append(f"  - First seen: {gap.first_seen}  Last seen: {gap.last_seen}")
                lines.append("")
    return lines


def _render_c5_contradictions(findings: LintFindings, summary: LintSummary) -> list[str]:
    """C5 Contradictions — always rendered (empty-section convention)."""
    n_c5 = len(findings.page_pairs)
    lines: list[str] = [_check_heading("C5", f"Contradictions ({n_c5} findings)"), ""]
    # Honesty note for the similarity cap (issue #194): when candidate pairs
    # exceed KB_LINT_C5_MAX_PAIRS, only the top-K most-similar are judged. Surface
    # the remainder so a capped audit reads as partial-by-design, not silent.
    if summary.c5_pairs_capped > 0:
        lines.append(
            f"> Judged the {summary.llm_calls} most-similar candidate page-pair(s); "
            f"{summary.c5_pairs_capped} further pair(s) were **not judged (capped** by "
            f"`KB_LINT_C5_MAX_PAIRS`). Raise the cap to audit more pairs."
        )
        lines.append("")
    if not findings.page_pairs:
        lines.append("_No page-pair contradictions found._")
        lines.append("")
    else:
        # Sub-group by severity in fixed order: direct → tension → duplicate
        by_sev: dict[str, list[PagePairFinding]] = defaultdict(list)
        for ppf in findings.page_pairs:
            by_sev[ppf.severity].append(ppf)

        for sev in ("direct", "tension", "duplicate"):
            group = by_sev.get(sev, [])
            if not group:
                continue
            lines.append(f"#### {sev.capitalize()} ({len(group)})")
            lines.append("")
            lines.append("| Page A | Page B | Page A claim | Page B claim | Suggested action |")
            lines.append("|--------|--------|-------------|-------------|------------------|")
            for ppf in group:
                claim_a = ppf.page_a_claim.replace("|", "\\|").replace("\n", " ")[:80]
                claim_b = ppf.page_b_claim.replace("|", "\\|").replace("\n", " ")[:80]
                action = ppf.suggested_action.replace("|", "\\|").replace("\n", " ")[:80]
                lines.append(
                    f"| `{ppf.page_a}` | `{ppf.page_b}` | {claim_a} | {claim_b} | {action} |"
                )
            lines.append("")
            for ppf in group:
                lines.append(f"**`{ppf.page_a}` ↔ `{ppf.page_b}`** — {ppf.summary}")
                lines.append("")
    return lines


def _render_c8_promotion_candidates(findings: LintFindings) -> list[str]:
    """C8 Promotion Candidates — omitted entirely when empty (Slice 6-5 convention:
    keeps noise out of the report while the qa lifecycle is dormant)."""
    if not findings.promotion_candidates:
        return []
    lines: list[str] = [_check_heading("C8", "Promotion Candidates"), ""]
    lines.append("| Slug | Question | Count | Age (days) | Cited |")
    lines.append("|------|----------|-------|------------|-------|")
    for pc in findings.promotion_candidates:
        q_cell = pc.question.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{pc.slug}` | {q_cell} | {pc.count} | {pc.age_days:.1f} | {pc.cited_count} |"
        )
    lines.append("")
    return lines


def _render_c9_stale_filed_answers(findings: LintFindings) -> list[str]:
    """C9 Stale Filed Answers — omitted entirely when empty (Slice 6-5 convention)."""
    if not findings.stale_filed_answers:
        return []
    lines: list[str] = [_check_heading("C9", "Stale Filed Answers"), ""]
    lines.append("| Slug | Stale Entity Citations | Days Drift |")
    lines.append("|------|------------------------|------------|")
    for sf in findings.stale_filed_answers:
        cites_cell = ", ".join(f"`{c}`" for c in sf.stale_citations)
        lines.append(f"| `{sf.page_slug}` | {cites_cell} | {sf.max_drift_days:.1f} |")
    lines.append("")
    return lines


def _render_c10_invalid_qa_schemas(findings: LintFindings) -> list[str]:
    """C10 Invalid qa Schema — omitted entirely when empty (Slice 6-5 convention)."""
    if not findings.invalid_qa_schemas:
        return []
    lines: list[str] = [_check_heading("C10", "Invalid qa Schema"), ""]
    lines.append("| Slug | Property | Offending Value |")
    lines.append("|------|----------|-----------------|")
    for inv in findings.invalid_qa_schemas:
        val_cell = inv.offending_value.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{inv.page_slug}` | `{inv.property_name}` | {val_cell} |")
    lines.append("")
    return lines


def _render_report_markdown(
    findings: LintFindings,
    summary: LintSummary,
    check_errors: dict[str, str],
) -> str:
    """Render the human-readable lint report as a markdown string.

    Sections are grouped under four axis ``## <Axis>`` headers (issue #361 /
    CONTEXT.md "Lint Axis"), in ``LINT_AXIS_ORDER`` (Freshness -> Coherence ->
    Coverage -> Lifecycle); within an axis, checks appear in
    ``LINT_CHECK_TAXONOMY``'s per-axis order via ``group_findings_by_axis``,
    each as its own ``### <CODE> ... — <label>`` sub-heading (built by
    ``_check_heading``). Each check keeps its own empty-section convention:
    C1/C2/C3/C4/C5/C6/C11 always render (with a "_No … found._" placeholder
    when empty), while C8/C9/C10 (Slice 6-5) are omitted entirely when empty.

    The report also starts with the sentinel HTML comment
    ``<!-- Auto-generated by POST /lint``, a ``# Lint Report`` heading, and a
    summary blockquote — all ahead of the axis sections.
    """
    lines: list[str] = []

    lines.append("<!-- Auto-generated by POST /lint — manual edits will be overwritten. -->")
    lines.append("")
    lines.append("# Lint Report")
    lines.append("")
    lines.append(
        f"> Generated at {summary.generated_at} · total findings: {summary.total_findings}"
    )
    lines.append("")

    # code -> its rendered lines (including the check's own H3 heading), built
    # once up front so the axis loop below only decides ordering, not content.
    check_lines_by_code: dict[str, list[str]] = {
        "C11": _render_c11_orphans(findings),
        "C3": _render_c3_failed_grounding(findings),
        "C4": _render_c4_slug_collisions(findings),
        "C6": _render_c6_stale_pages(findings),
        "C2": _render_c2_red_links(findings),
        "C1": _render_c1_coverage_gaps(findings),
        "C5": _render_c5_contradictions(findings, summary),
        "C8": _render_c8_promotion_candidates(findings),
        "C9": _render_c9_stale_filed_answers(findings),
        "C10": _render_c10_invalid_qa_schemas(findings),
    }

    for axis_group in group_findings_by_axis(findings):
        # Build the axis body first so an all-empty axis skips its ``## <Axis>``
        # header entirely rather than emitting a dangling heading with nothing
        # beneath it. Only Lifecycle is elidable in practice: its checks
        # (C8/C9/C10) self-omit when empty, so a dormant qa lifecycle — the
        # common case — would otherwise render a bare ``## Lifecycle``. The
        # Freshness/Coherence/Coverage checks always render a "_No … found._"
        # placeholder, so those axes are never empty (issue #361).
        axis_body: list[str] = []
        for meta, _findings_list in axis_group.checks:
            axis_body.extend(check_lines_by_code[meta.code])
        if not axis_body:
            continue
        lines.append(f"## {axis_group.axis}")
        lines.append("")
        lines.extend(axis_body)

    if check_errors:
        lines.append("")
        lines.append("## Check errors")
        lines.append("")
        for check_id, err_msg in check_errors.items():
            lines.append(f"- **{check_id}**: {err_msg}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Report writer (atomic)
# ---------------------------------------------------------------------------


def _write_report(report_path: Path, content: str) -> None:
    """Write lint-report.md atomically via the shared write_text_atomic helper (§2.6)."""
    write_text_atomic(report_path, content)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def run_lint(
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
    log_path: Path | None = None,
    include_c5: bool = True,
) -> LintResponse:
    """Run the lint checks; write wiki/lint-report.md; return LintResponse.

    Read-only with respect to wiki page frontmatter.
    Holds ``indexer._index_lock`` for the full duration.
    Continue-on-error: a check that raises is recorded in
    ``LintResponse.check_errors``; other checks still run.

    Parameters default to the module-level constants (``WIKI_DIR``, ``DOCS_DIR``,
    ``LOG_PATH``) so tests can monkeypatch those attributes without passing kwargs.

    ``include_c5`` (default True) controls the only LLM-backed check, C5
    (page-pair contradiction detection). C5 makes one LLM call per candidate
    page-pair, so it dominates lint wall-time and cost on a large wiki. Callers
    that only need the fast, local checks — notably the Console's Curation
    Queue, which reads C8/C9/C10 — pass ``include_c5=False`` to skip it; the
    response then carries ``page_pairs == []`` and ``llm_calls == 0``.

    Returns a LintResponse (which is also JSON-serialisable via FastAPI).
    """
    resolved_wiki = wiki_dir if wiki_dir is not None else WIKI_DIR
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR
    resolved_log = log_path if log_path is not None else LOG_PATH

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_event("lint_started", "", log_path=resolved_log)

    check_errors: dict[str, str] = {}

    # Hold the indexer lock for the entire check sequence so /ingest cannot mutate
    # wiki pages mid-snapshot. Lint is read-only, so the lock is purely a
    # consistency guard, not a write barrier on lint's side.
    with _index_lock:
        # --- C11 Orphan pages ---
        orphans: list[OrphanPageFinding] = []
        try:
            orphans = _check_c11_orphan(resolved_wiki, resolved_docs)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c11"] = err_msg
            log_event("lint_check_error", f"check=c11 exc={err_msg}", log_path=resolved_log)

        # --- C3 Failed-grounding sweep ---
        failed_grounding: list[FailedGroundingFinding] = []
        try:
            failed_grounding = _check_c3_failed_grounding(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c3"] = err_msg
            log_event("lint_check_error", f"check=c3 exc={err_msg}", log_path=resolved_log)

        # --- C4-a Slug collision groups ---
        slug_collisions: list[SlugCollisionFinding] = []
        try:
            slug_collisions = _check_c4a_slug_collision(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c4a"] = err_msg
            log_event("lint_check_error", f"check=c4a exc={err_msg}", log_path=resolved_log)

        # --- C6 Stale pages ---
        stale_pages: list[StalePageFinding] = []
        try:
            stale_pages = _check_c6_stale(resolved_wiki, resolved_docs)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c6"] = err_msg
            log_event("lint_check_error", f"check=c6 exc={err_msg}", log_path=resolved_log)

        # --- C2 Red links ---
        red_links: list[RedLinkFinding] = []
        try:
            red_links = _check_c2_red_links(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c2"] = err_msg
            log_event("lint_check_error", f"check=c2 exc={err_msg}", log_path=resolved_log)

        # --- C1 Coverage gaps ---
        coverage_gaps: list[CoverageGapFinding] = []
        try:
            coverage_gaps = _check_c1_coverage_gaps(resolved_log)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c1"] = err_msg
            log_event("lint_check_error", f"check=c1 exc={err_msg}", log_path=resolved_log)

        # --- C8 Promotion candidates (read-only qa scan) ---
        promotion_candidates: list[PromotionCandidateFinding] = []
        try:
            promotion_candidates = _check_c8_promotion_candidates(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c8"] = err_msg
            log_event("lint_check_error", f"check=c8 exc={err_msg}", log_path=resolved_log)

        # --- C9 qa-staleness (read-only entity-vs-qa mtime comparison) ---
        stale_filed_answers: list[QaStalenessFinding] = []
        try:
            stale_filed_answers = _check_c9_qa_staleness(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c9"] = err_msg
            log_event("lint_check_error", f"check=c9 exc={err_msg}", log_path=resolved_log)

        # --- C10 qa-schema-validity (read-only schema sweep) ---
        invalid_qa_schemas: list[InvalidQaSchemaFinding] = []
        try:
            invalid_qa_schemas = _check_c10_qa_schema_validity(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c10"] = err_msg
            log_event("lint_check_error", f"check=c10 exc={err_msg}", log_path=resolved_log)

        # --- C5 Page-pair contradiction detection (most expensive — last) ---
        # The only LLM-backed check: one call per candidate page-pair, so it
        # dominates wall-time and cost on a large wiki. Skipped wholesale when
        # include_c5 is False (fast path for the Curation Queue) — page_pairs,
        # llm_calls and cost_usd then keep their zero defaults below.
        page_pairs: list[PagePairFinding] = []
        llm_calls: int = 0
        cost_usd: float = 0.0
        c5_pairs_capped: int = 0
        if include_c5:
            try:
                page_pairs = _check_c5_page_pair(resolved_wiki)
                # _check_c5_page_pair records exact run metrics in module-level
                # counters (one LLM call per judged pair, capped at
                # KB_LINT_C5_MAX_PAIRS): llm_calls is the judged count and
                # c5_pairs_capped the not-judged remainder. Read then reset both
                # so a later run in the same process starts clean.
                llm_calls = _c5_llm_call_counter[0]
                c5_pairs_capped = _c5_capped_counter[0]
                # gpt-4o-mini pricing (2025): $0.15/1M input tokens, $0.60/1M output tokens.
                # Rough estimate ~500 input + 150 output tokens/call ≈ $0.000165/call.
                cost_usd = round(llm_calls * 0.000165, 6)
                _c5_llm_call_counter[0] = 0
                _c5_capped_counter[0] = 0
                # Per-pair LLM errors are caught inside _check_c5_page_pair
                # (continue-on-error) but surfaced here in check_errors["c5"]
                # so the SUCCESS payload reflects the partial failure.
                # server.py docstring: "Individual per-pair LLM errors within C5
                # are NOT isError — they are recorded in check_errors['c5']."
                if _c5_pair_errors:
                    check_errors["c5"] = "; ".join(e[:200] for e in _c5_pair_errors)
                    _c5_pair_errors.clear()
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                check_errors["c5"] = err_msg
                log_event("lint_check_error", f"check=c5 exc={err_msg}", log_path=resolved_log)

    # --- Aggregate findings ---
    findings = LintFindings(
        orphans=orphans,
        failed_grounding=failed_grounding,
        slug_collisions=slug_collisions,
        stale_pages=stale_pages,
        red_links=red_links,
        coverage_gaps=coverage_gaps,
        page_pairs=page_pairs,
        promotion_candidates=promotion_candidates,
        stale_filed_answers=stale_filed_answers,
        invalid_qa_schemas=invalid_qa_schemas,
    )
    total = (
        len(orphans)
        + len(failed_grounding)
        + len(slug_collisions)
        + len(stale_pages)
        + len(red_links)
        + len(coverage_gaps)
        + len(page_pairs)
        + len(promotion_candidates)
        + len(stale_filed_answers)
        + len(invalid_qa_schemas)
    )
    findings_by_check: dict[str, int] = {
        "c11": len(orphans),
        "c3": len(failed_grounding),
        "c4a": len(slug_collisions),
        "c6": len(stale_pages),
        "c2": len(red_links),
        "c1": len(coverage_gaps),
        "c5": len(page_pairs),
        "c8": len(promotion_candidates),
        "c9": len(stale_filed_answers),
        "c10": len(invalid_qa_schemas),
    }

    summary = LintSummary(
        total_findings=total,
        findings_by_check=findings_by_check,
        llm_calls=llm_calls,
        cost_usd=cost_usd,
        c5_pairs_capped=c5_pairs_capped,
        generated_at=generated_at,
    )

    # --- Write report ---
    report_path = resolved_wiki / "lint-report.md"
    report_content = _render_report_markdown(findings, summary, check_errors)
    _write_report(report_path, report_content)

    # --- Log completed ---
    by_check_str = ",".join(f"{k}:{v}" for k, v in findings_by_check.items())
    log_event(
        "lint_completed",
        f"findings={total} by_check={by_check_str} llm_calls={llm_calls} cost_usd={cost_usd:.6f} errors={len(check_errors)}",
        log_path=resolved_log,
    )

    return LintResponse(
        report_path=str(report_path.relative_to(resolved_wiki.parent))
        if report_path.is_relative_to(resolved_wiki.parent)
        else str(report_path),
        findings=findings,
        summary=summary,
        check_errors=check_errors,
    )
