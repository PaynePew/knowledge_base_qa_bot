"""Deep module per Ousterhout. Public surface: ``build_spotcheck_subset``, ``run_spotcheck``, ``SpotcheckItem``, ``SpotcheckResult``, ``JudgeUnavailableError``, ``DEFAULT_JUDGE_MODEL``, ``JUDGE_MODELS``, ``ZONE_*``.

L2 cross-family LLM-judge **Spot-check** for the Phase 8 retrieval comparison
(CONTEXT.md § Phase 8 > Spot-check, PRD #100 user stories 18-21, issue #105).

The deterministic L1 (C5c) metric is the source of every headline number. The
Spot-check is an OPT-IN second opinion that re-judges L1's *edge-case* verdicts
using a judge from a DIFFERENT model family (Claude) than the OpenAI embedding
that powers Stack B — so the judge cannot share a same-family blind spot with
the arm it is checking. The Spot-check produces NO headline numbers; it reports
only a by-zone agreement rate against L1.

The Spot-check input is an **ambiguous subset** — the union of three zones over
the per-Paraphrase, per-Stack L1 verdicts:

  1. **Marginal** (``ZONE_MARGINAL``) — the retrieved top-1 item's source matches
     the Gold Section AND its content shares only ``≤ marginal_threshold`` Key
     Tokens (a "barely a hit" the metric is least sure about).
  2. **Disagreement** (``ZONE_DISAGREEMENT``) — Stack A's top-1 L1 verdict differs
     from Stack B's top-1 L1 verdict for the same Paraphrase (the two arms
     disagree, so at most one is right).
  3. **Control** (``ZONE_CONTROL``) — a seeded random sample of clear hits and
     clear misses where L1 is confident; the judge's agreement here must approach
     100% or the judge itself is mis-calibrated and its other verdicts are
     suspect (PRD #100 user story 21).

Determinism: zone membership is a pure function of the L1 verdicts (no model
call), and the Control sample is drawn with a seeded ``random.Random`` so the
subset is reproducible across runs and tests.

§2.4 isolation: the Anthropic client and its message/response types live ENTIRELY
inside this module. ``run_spotcheck`` returns only Python primitives — verdicts
(bool), reasoning (str), and aggregated agreement rates (float) — so the report
and metric modules never import or see an Anthropic type. The judge client is
constructed lazily via ``_judge_client`` (a getter the tests stub), mirroring the
lazy-singleton LLM pattern (CODING_STANDARD §2.7 / §10).
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from markdown_kb.app.indexer import tokenize

from .metric import DEFAULT_K, is_hit
from .models import Paraphrase, RetrievedItem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The documented judge choices (issue #105). Default is Sonnet — the mid-tier
# cross-family judge; haiku is cheaper, opus is the strongest reasoner.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MODELS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
)

# Zone identifiers (the three ambiguous-subset zones, PRD #100 user story 20).
ZONE_MARGINAL = "marginal"
ZONE_DISAGREEMENT = "disagreement"
ZONE_CONTROL = "control"
ZONES: tuple[str, ...] = (ZONE_MARGINAL, ZONE_DISAGREEMENT, ZONE_CONTROL)

DEFAULT_MARGINAL_THRESHOLD = 1
DEFAULT_CONTROL_SAMPLE_SIZE = 5
# Seeded so the Control sample (and therefore the whole subset) is reproducible.
_CONTROL_SEED = 42

# Bounded, grep-friendly judge prompt. The judge decides a single yes/no: does
# the retrieved CONTENT actually answer the Paraphrase? It never sees the L1
# verdict, so its agreement with L1 is an independent signal.
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict retrieval-quality judge. You are given a user QUESTION and "
    "a passage of retrieved CONTENT. Decide ONLY whether the CONTENT actually "
    "answers the QUESTION — not whether it is on a vaguely related topic. Reply "
    "with a JSON object {\"answers\": true|false, \"reasoning\": \"<one sentence>\"}."
)
_JUDGE_MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class JudgeUnavailableError(RuntimeError):
    """Raised (fail-fast) when ``--judge`` is set but ``ANTHROPIC_API_KEY`` is absent.

    The Spot-check is opt-in; a caller who explicitly asks for it must get a
    clear, immediate error rather than a silent skip or a deep SDK auth failure
    (CODING_STANDARD §4.1 fail-fast).
    """


# ---------------------------------------------------------------------------
# Data model (primitives only — no Anthropic types cross this boundary)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpotcheckItem:
    """One ambiguous-subset member: a (Paraphrase, Stack) pair the judge re-judges.

    ``l1_hit`` is the deterministic C5c verdict for this Stack's top-1 retrieval
    of the Paraphrase; the judge's independent verdict is compared against it to
    compute agreement. ``content`` is the retrieved text the judge reads. An item
    may belong to more than one zone (e.g. a marginal hit that is also a
    disagreement); ``zones`` carries every zone it qualified for.
    """

    paraphrase_id: str
    stack: str
    zones: tuple[str, ...]
    query: str
    content: str
    l1_hit: bool


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's independent answers-the-question call for one item (primitives)."""

    answers: bool
    reasoning: str


@dataclass(frozen=True)
class SpotcheckResult:
    """Aggregated Spot-check outcome — primitives only, for the report renderer.

    ``judge_model`` records which Claude model ran. ``subset_size_by_zone`` and
    ``agreement_by_zone`` are keyed by the ``ZONE_*`` ids; ``agreement_by_zone``
    is the fraction of items in that zone whose judge verdict matched the L1
    verdict. ``items`` carries the per-item verdicts + reasoning so the report can
    surface examples. The report reads ONLY this dataclass — it never imports
    anthropic.
    """

    judge_model: str
    marginal_threshold: int
    control_sample_size: int
    zones_requested: tuple[str, ...]
    subset_size_by_zone: dict[str, int]
    agreement_by_zone: dict[str, float]
    verdicts: list[tuple[SpotcheckItem, JudgeVerdict]] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(self.subset_size_by_zone.values())


# A retrieval entry point: ``(query, k) -> list[RetrievedItem]`` (matches stacks).
StackRetrieval = Callable[[str, int], list[RetrievedItem]]


# ---------------------------------------------------------------------------
# Per-Paraphrase L1 verdict capture (the deterministic input to zone selection)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ParaphraseVerdict:
    """Per-Paraphrase, per-Stack top-1 L1 facts the zones are built from.

    For each Stack: the top-1 retrieved item (or None if nothing retrieved), its
    deterministic top-1 ``is_hit`` verdict, and the count of Key Tokens its
    content overlaps (used by the Marginal-zone threshold test).
    """

    paraphrase: Paraphrase
    a_item: RetrievedItem | None
    b_item: RetrievedItem | None
    a_hit: bool
    b_hit: bool
    a_overlap: int
    b_overlap: int


def _overlap_count(item: RetrievedItem | None, key_tokens: set[str]) -> int:
    if item is None:
        return 0
    return len(set(tokenize(item.content)) & key_tokens)


def _capture_verdicts(
    paraphrases: list[Paraphrase],
    retrieve_a: StackRetrieval,
    retrieve_b: StackRetrieval,
    k: int,
) -> list[_ParaphraseVerdict]:
    """Run both Stacks once per Paraphrase and record the top-1 L1 facts.

    Pure deterministic capture — no judge call. The retrieval callables are the
    same in-process Stack adapters the runner uses, so the zones are built from
    the identical L1 verdicts that produce the headline numbers.
    """
    verdicts: list[_ParaphraseVerdict] = []
    for para in paraphrases:
        gold = para.gold_docs_section_id
        wanted = para.key_tokens
        a_items = retrieve_a(para.text, k)
        b_items = retrieve_b(para.text, k)
        a_top = a_items[0] if a_items else None
        b_top = b_items[0] if b_items else None
        verdicts.append(
            _ParaphraseVerdict(
                paraphrase=para,
                a_item=a_top,
                b_item=b_top,
                a_hit=bool(a_top and is_hit(a_top, gold, wanted)),
                b_hit=bool(b_top and is_hit(b_top, gold, wanted)),
                a_overlap=_overlap_count(a_top, wanted),
                b_overlap=_overlap_count(b_top, wanted),
            )
        )
    return verdicts


# ---------------------------------------------------------------------------
# Zone selection (deterministic, pure over the captured verdicts)
# ---------------------------------------------------------------------------
def build_spotcheck_subset(
    paraphrases: list[Paraphrase],
    retrieve_a: StackRetrieval,
    retrieve_b: StackRetrieval,
    *,
    k: int = DEFAULT_K,
    zones: tuple[str, ...] = ZONES,
    marginal_threshold: int = DEFAULT_MARGINAL_THRESHOLD,
    control_sample_size: int = DEFAULT_CONTROL_SAMPLE_SIZE,
    seed: int = _CONTROL_SEED,
) -> list[SpotcheckItem]:
    """Assemble the ambiguous subset (the union of the requested zones).

    The subset is built from the deterministic L1 verdicts (see
    ``_capture_verdicts``); no judge is called here. Rules:

      - **Marginal**: a Stack's top-1 item matches the Gold Section AND its
        Key-Token overlap is ``1 ≤ overlap ≤ marginal_threshold`` (a hit the
        metric is least sure about — overlap 0 is a clear miss, not marginal).
      - **Disagreement**: Stack A's top-1 hit verdict ≠ Stack B's top-1 hit
        verdict for the same Paraphrase. Both Stacks' items enter the subset.
      - **Control**: a ``seed``-seeded sample of ``control_sample_size`` clear
        hits (top-1 hit with overlap > marginal_threshold) and the same number of
        clear misses (top-1 not a hit), so the judge baseline can be confirmed.

    Items are de-duplicated by (paraphrase_id, stack); an item qualifying for
    several zones carries all of them in ``zones`` (sorted). The returned order
    is deterministic.
    """
    requested = tuple(z for z in zones if z in ZONES)
    verdicts = _capture_verdicts(paraphrases, retrieve_a, retrieve_b, k)

    # Accumulate zone membership per (paraphrase_id, stack) so an item that
    # qualifies for multiple zones is judged once but counts toward each.
    by_key: dict[tuple[str, str], dict] = {}

    def _add(v: _ParaphraseVerdict, stack: str, item: RetrievedItem, hit: bool, zone: str):
        key = (v.paraphrase.paraphrase_id, stack)
        entry = by_key.setdefault(
            key,
            {
                "query": v.paraphrase.text,
                "content": item.content,
                "l1_hit": hit,
                "zones": set(),
            },
        )
        entry["zones"].add(zone)

    if ZONE_MARGINAL in requested:
        for v in verdicts:
            if v.a_item and v.a_hit and 1 <= v.a_overlap <= marginal_threshold:
                _add(v, "Stack A", v.a_item, True, ZONE_MARGINAL)
            if v.b_item and v.b_hit and 1 <= v.b_overlap <= marginal_threshold:
                _add(v, "Stack B", v.b_item, True, ZONE_MARGINAL)

    if ZONE_DISAGREEMENT in requested:
        for v in verdicts:
            if v.a_hit != v.b_hit:
                if v.a_item is not None:
                    _add(v, "Stack A", v.a_item, v.a_hit, ZONE_DISAGREEMENT)
                if v.b_item is not None:
                    _add(v, "Stack B", v.b_item, v.b_hit, ZONE_DISAGREEMENT)

    if ZONE_CONTROL in requested:
        for v, stack, item, hit in _control_items(
            verdicts, marginal_threshold, control_sample_size, seed
        ):
            _add(v, stack, item, hit, ZONE_CONTROL)

    items = [
        SpotcheckItem(
            paraphrase_id=pid,
            stack=stack,
            zones=tuple(sorted(entry["zones"])),
            query=entry["query"],
            content=entry["content"],
            l1_hit=entry["l1_hit"],
        )
        for (pid, stack), entry in by_key.items()
    ]
    # Stable, deterministic order: by paraphrase id then stack.
    items.sort(key=lambda it: (it.paraphrase_id, it.stack))
    return items


def _control_items(
    verdicts: list[_ParaphraseVerdict],
    marginal_threshold: int,
    sample_size: int,
    seed: int,
) -> list[tuple[_ParaphraseVerdict, str, RetrievedItem, bool]]:
    """Seeded clear-hit + clear-miss control sample (deterministic).

    A clear hit is a top-1 hit whose Key-Token overlap exceeds the marginal
    threshold (so it is NOT a marginal edge case); a clear miss is a top-1 that
    is not a hit. We sample ``sample_size`` of each from Stack B (the arm L1 is
    documented to over-estimate, PRD disclosure 5), using a seeded ``Random`` over
    a stably-sorted candidate list so the sample is reproducible.
    """
    rng = random.Random(seed)
    clear_hits = [
        v for v in verdicts
        if v.b_item is not None and v.b_hit and v.b_overlap > marginal_threshold
    ]
    clear_misses = [v for v in verdicts if v.b_item is not None and not v.b_hit]
    clear_hits.sort(key=lambda v: v.paraphrase.paraphrase_id)
    clear_misses.sort(key=lambda v: v.paraphrase.paraphrase_id)

    chosen_hits = _sample(rng, clear_hits, sample_size)
    chosen_misses = _sample(rng, clear_misses, sample_size)

    out: list[tuple[_ParaphraseVerdict, str, RetrievedItem, bool]] = []
    for v in chosen_hits:
        out.append((v, "Stack B", v.b_item, True))  # type: ignore[arg-type]
    for v in chosen_misses:
        out.append((v, "Stack B", v.b_item, False))  # type: ignore[arg-type]
    return out


def _sample(rng: random.Random, pool: list, n: int) -> list:
    """Deterministically pick up to ``n`` items from a stably-sorted ``pool``."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# Judge call (Anthropic client + types isolated entirely below this line)
# ---------------------------------------------------------------------------
def run_spotcheck(
    paraphrases: list[Paraphrase],
    retrieve_a: StackRetrieval,
    retrieve_b: StackRetrieval,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    k: int = DEFAULT_K,
    zones: tuple[str, ...] = ZONES,
    marginal_threshold: int = DEFAULT_MARGINAL_THRESHOLD,
    control_sample_size: int = DEFAULT_CONTROL_SAMPLE_SIZE,
    seed: int = _CONTROL_SEED,
) -> SpotcheckResult:
    """Run the opt-in L2 Spot-check and return primitive-only aggregates.

    Fail-fast (``JudgeUnavailableError``) if ``ANTHROPIC_API_KEY`` is absent — the
    caller explicitly opted in via ``--judge``. Builds the ambiguous subset
    (deterministic), asks the cross-family Claude judge whether each item's
    retrieved content answers the Paraphrase, and aggregates the per-zone
    agreement rate against the deterministic L1 verdict.

    The returned ``SpotcheckResult`` carries only primitives; no Anthropic type
    escapes this function (CODING_STANDARD §2.4).
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise JudgeUnavailableError(
            "--judge requires ANTHROPIC_API_KEY to be set (the L2 Spot-check calls "
            "the Claude judge). Set ANTHROPIC_API_KEY, or omit --judge to skip the "
            "opt-in cross-family validation and report L1 numbers only."
        )

    subset = build_spotcheck_subset(
        paraphrases,
        retrieve_a,
        retrieve_b,
        k=k,
        zones=zones,
        marginal_threshold=marginal_threshold,
        control_sample_size=control_sample_size,
        seed=seed,
    )

    client = _judge_client()
    verdicts: list[tuple[SpotcheckItem, JudgeVerdict]] = []
    for item in subset:
        verdicts.append((item, _judge_one(client, judge_model, item)))

    return _aggregate(
        judge_model=judge_model,
        marginal_threshold=marginal_threshold,
        control_sample_size=control_sample_size,
        zones_requested=tuple(z for z in zones if z in ZONES),
        verdicts=verdicts,
    )


def _aggregate(
    judge_model: str,
    marginal_threshold: int,
    control_sample_size: int,
    zones_requested: tuple[str, ...],
    verdicts: list[tuple[SpotcheckItem, JudgeVerdict]],
) -> SpotcheckResult:
    """Fold per-item verdicts into by-zone subset sizes + agreement rates.

    Agreement for an item is ``judge.answers == item.l1_hit``. An item in N zones
    contributes to all N zones' rates, so the per-zone numbers stay independent.
    """
    size_by_zone: dict[str, int] = {z: 0 for z in zones_requested}
    agree_by_zone: dict[str, list[bool]] = {z: [] for z in zones_requested}
    for item, verdict in verdicts:
        agrees = verdict.answers == item.l1_hit
        for zone in item.zones:
            if zone not in size_by_zone:
                size_by_zone[zone] = 0
                agree_by_zone[zone] = []
            size_by_zone[zone] += 1
            agree_by_zone[zone].append(agrees)
    agreement = {
        z: (sum(flags) / len(flags) if flags else 0.0)
        for z, flags in agree_by_zone.items()
    }
    return SpotcheckResult(
        judge_model=judge_model,
        marginal_threshold=marginal_threshold,
        control_sample_size=control_sample_size,
        zones_requested=zones_requested,
        subset_size_by_zone=size_by_zone,
        agreement_by_zone=agreement,
        verdicts=verdicts,
    )


# ---------------------------------------------------------------------------
# Anthropic adapter (the ONLY code that touches anthropic types) — §2.4
# ---------------------------------------------------------------------------
def _judge_client():
    """Lazily construct the Anthropic client (lazy-singleton getter; §2.7).

    Function-scope import keeps the anthropic dependency eval-time only and lets
    tests stub this getter without importing anthropic. The constructed client
    type never escapes this module.
    """
    import anthropic  # eval-time-only judge dep; isolated here (§2.4)

    return anthropic.Anthropic()


def _judge_one(client, judge_model: str, item: SpotcheckItem) -> JudgeVerdict:
    """Ask the Claude judge a single answers-the-question call; return primitives.

    The Anthropic request/response objects are consumed entirely here — only the
    parsed ``JudgeVerdict`` (bool + str) leaves this function, so no SDK type
    leaks to the runner/report (§2.4).
    """
    user = (
        f"QUESTION:\n{item.query}\n\n"
        f"CONTENT:\n{item.content}\n\n"
        'Does the CONTENT answer the QUESTION? Reply with the JSON object only.'
    )
    response = client.messages.create(
        model=judge_model,
        max_tokens=_JUDGE_MAX_TOKENS,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return _parse_verdict(_response_text(response))


def _response_text(response) -> str:
    """Flatten an Anthropic ``Message`` content list to its text (isolated here)."""
    parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _parse_verdict(text: str) -> JudgeVerdict:
    """Parse the judge's JSON reply into a primitive ``JudgeVerdict``.

    Tolerant of code-fence wrapping; on an unparseable reply the verdict defaults
    to ``answers=False`` with the raw text as reasoning (a fail-safe that surfaces
    the parse problem in the report rather than crashing the run).
    """
    import json

    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
        return JudgeVerdict(
            answers=bool(data.get("answers", False)),
            reasoning=str(data.get("reasoning", "")).strip(),
        )
    except (ValueError, AttributeError):
        return JudgeVerdict(answers=False, reasoning=f"unparseable judge reply: {text[:120]}")
