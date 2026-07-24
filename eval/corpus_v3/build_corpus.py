"""Deep module per Ousterhout. Public surface: ``AdversarialGroup``,
``RawSection``, ``ADVERSARIAL_GROUPS``, ``write_corpus_fixtures``,
``write_build_cost_report``.

Seeded, deterministic build script for the corpus v3 adversarial corpus
fixtures (issue #661, PRD #654 user stories 10-12 / 18, ADR-0045's "curated
layer's claimed home ground"). Every raw Source and curated wiki page this
script writes is committed to git; this script is how they are
**regenerated**, not how they are served — nothing under ``eval/corpus_v3``
reads through this module at test/query time (mirrors
``eval.paraphrase_comparison.build_wiki_fixtures``'s "one-off CLI to
(re)build committed fixtures" role).

``ADVERSARIAL_GROUPS`` is the single source of truth (mirrors
``eval.paraphrase_comparison.generation.corpus_generator.DOC_SPECS``): a
static Python data structure, not prose, so the corpus can be regenerated
byte-for-byte and extended by appending an entry rather than hand-editing
markdown. Three adversarial classes, matching the issue's "what to build":

- ``redundancy``   — two-or-more near-duplicate raw Sources describing the
  SAME fact in different wording; the curated wiki page cites BOTH
  (``sources:`` 1:N) and states one canonical answer, demonstrating the
  layer's dedup value (a raw-docs stack has no such consolidation and may
  waste top-k slots on both near-duplicates).
- ``contradiction`` — two raw Sources that directly conflict on a fact; the
  curated wiki page cites BOTH but does NOT pick a winner — the conflict is
  recorded in ``open_questions`` instead, so this fixture set never asserts
  a false single truth. This is the ``contradiction-leak rate`` axis's home
  ground (ADR-0045): whether a stack's answer cites the wiki's honest
  "unresolved" framing or leaks one of the raw conflicting figures.
- ``version_evolution`` — three dated raw Sources for the same fact, each
  superseding the last; the curated wiki page cites ONLY the newest
  (``sources:`` 1:1 to the latest id), giving version-conflict queries
  (``query_schema.ScenarioStratum`` = ``"version_conflict"``) a defined gold
  answer that a raw-docs stack (no curation, all three versions equally
  retrievable) does not have.

Construction method and the cost-ledger AC: every group here is
hand-authored directly into ``ADVERSARIAL_GROUPS`` — no LLM synthesis call
is made to produce it (unlike ``build_wiki_fixtures.py``'s docs-to-wiki
paraphrase, there is no "reword this existing Source" step; each side of a
redundancy/contradiction/version group is an independent authored fact).
:func:`write_corpus_fixtures` still records one zero-usage ``CostLedger``
entry per group under phase ``"build"`` — an honest, real (not fabricated)
zero, not a live LLM cost — so the ledger accurately reflects "offline,
deterministic construction" as the phase's actual cost, the same contingency
``build_wiki_fixtures.py`` documents for its own no-API-key fallback. A
future live corpus-build run (synthesis via a real LLM) would record real
non-zero entries into the same ledger shape via ``eval.cost_ledger.hooks``.

Query-set-size reconciliation (issue #661 AC 2, POWER_ANALYSIS.md's
"Sensitivity" section): the power analysis (#660) derives n=909 fully-powered
English queries per SCENARIO stratum, not per adversarial CORPUS instance —
query generation (a later, still-unbuilt issue per ``generation/SPEC.md``
"Out of scope for this issue (#660)") draws many queries per group via
paraphrase/multi-family variation. This module's 3-groups-per-class
minimum (below) is the corpus-side half of that reconciliation: enough
DISTINCT topical instances per class that a generator can multiply toward
the target n without repeating the same underlying fact pattern into
near-duplicate queries. See ``CORPUS.md`` for the full accounting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata
from markdown_kb.app.indexer import slugify

_PKG_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = _PKG_ROOT / "corpus"
WIKI_CONCEPTS_DIR = _PKG_ROOT / "wiki" / "concepts"

# Fixed build timestamp (matches the existing corpus v3 wiki fixtures'
# convention, e.g. wiki/concepts/password-reset.md) so a regeneration run is
# byte-identical to the committed fixtures rather than drifting on wall-clock
# time.
BUILD_TIMESTAMP = "2026-07-24T00:00:00Z"

AdversarialClass = Literal["redundancy", "contradiction", "version_evolution"]

# The minimum number of DISTINCT topical instances required per adversarial
# class (issue #661 AC 2) — enforced by a test, not just documented.
MIN_INSTANCES_PER_CLASS = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawSection:
    """One raw, docs-native Source file: one heading, one leaf Section body.

    Optional ``frontmatter`` is used only by ``version_evolution`` sections
    (an ``updated`` date and, for every version after the first, a
    ``supersedes`` pointer to the prior version's docs-native id) — the 11-rule
    Section parser (``markdown_kb.app.indexer.parse_markdown`` rule 2) accepts
    frontmatter on ANY Source, not just wiki pages, and never tokenizes it.
    """

    basename: str
    heading: str
    body: str
    frontmatter: dict | None = None

    @property
    def anchor(self) -> str:
        return slugify(self.heading)

    @property
    def source_id(self) -> str:
        return f"{self.basename}#{self.anchor}"


@dataclass(frozen=True)
class AdversarialGroup:
    """One adversarial instance: its raw Sections plus the one curated wiki
    concept page a real ``/ingest`` curation pass would produce over them."""

    adversarial_class: AdversarialClass
    group_id: str
    wiki_title: str
    wiki_body: str
    sections: list[RawSection]
    gold_section_ids: list[str]  # subset of `sections`' ids the wiki page cites
    open_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        basenames = {s.basename for s in self.sections}
        if len(basenames) != len(self.sections):
            raise ValueError(f"group {self.group_id!r}: duplicate basenames")
        ids = {s.source_id for s in self.sections}
        missing = set(self.gold_section_ids) - ids
        if missing:
            raise ValueError(
                f"group {self.group_id!r}: gold_section_ids not in sections: {missing}"
            )


# ---------------------------------------------------------------------------
# ADVERSARIAL_GROUPS — the single source of truth (3 classes x 3 instances)
# ---------------------------------------------------------------------------
ADVERSARIAL_GROUPS: list[AdversarialGroup] = [
    # ------------------------------------------------------------------ #
    # redundancy — near-duplicate Sources, curated page cites both
    # ------------------------------------------------------------------ #
    AdversarialGroup(
        adversarial_class="redundancy",
        group_id="store-hours",
        wiki_title="Store Hours",
        wiki_body=(
            "Acme Shop's physical storefront is open Monday through Friday, "
            "9:00 AM to 6:00 PM, and closed on public holidays. Two Source "
            "documents describe this policy in different wording; this page "
            "consolidates them into one canonical answer."
        ),
        sections=[
            RawSection(
                basename="store_hours_a.md",
                heading="Weekday Hours",
                body=(
                    "The Acme Shop storefront is open Monday through Friday, "
                    "9:00 AM to 6:00 PM. The store is closed on all public "
                    "holidays. Customers with questions outside these hours "
                    "can reach support by email."
                ),
            ),
            RawSection(
                basename="store_hours_b.md",
                heading="Weekday Hours",
                body=(
                    "Our physical location welcomes walk-in customers Monday "
                    "to Friday, from 9am until 6pm. We remain closed on "
                    "public holidays. For anything urgent outside business "
                    "hours, email is the fastest way to reach us."
                ),
            ),
        ],
        gold_section_ids=[
            "store_hours_a.md#weekday-hours",
            "store_hours_b.md#weekday-hours",
        ],
    ),
    AdversarialGroup(
        adversarial_class="redundancy",
        group_id="loyalty-signup",
        wiki_title="Loyalty Program Signup",
        wiki_body=(
            "Joining the loyalty program is free, takes about two minutes, "
            "and is done from account settings; point accrual begins on the "
            "next purchase. Two Source documents describe the same signup "
            "flow in different wording; this page consolidates them."
        ),
        sections=[
            RawSection(
                basename="loyalty_program.md",
                heading="How To Join",
                body=(
                    "Joining the loyalty program is free and takes about two "
                    "minutes. Create an account on the website, opt in to the "
                    "loyalty program from the account settings page, and "
                    "points begin accruing on your very next purchase."
                ),
            ),
            RawSection(
                basename="membership_faq.md",
                heading="Joining The Program",
                body=(
                    "Signing up for membership costs nothing and only takes a "
                    "couple of minutes. After creating an account, enable the "
                    "membership program under account settings; point accrual "
                    "starts with your next order."
                ),
            ),
        ],
        gold_section_ids=[
            "loyalty_program.md#how-to-join",
            "membership_faq.md#joining-the-program",
        ],
    ),
    AdversarialGroup(
        adversarial_class="redundancy",
        group_id="two-factor-setup",
        wiki_title="Two-Factor Setup",
        wiki_body=(
            "Two-factor authentication is enabled from Account > Security by "
            "scanning a QR code with an authenticator app; every sign-in "
            "afterward requires the six-digit code plus the password. Two "
            "Source documents describe this flow in different wording; this "
            "page consolidates them."
        ),
        sections=[
            RawSection(
                basename="account_security_2fa.md",
                heading="Two-Factor Setup",
                body=(
                    "Two-factor authentication adds a one-time code to the "
                    "sign-in process. Enable it from Account > Security by "
                    "scanning a QR code with an authenticator app. Once "
                    "enabled, every sign-in requires both the password and "
                    "the current six-digit code."
                ),
            ),
            RawSection(
                basename="security_faq.md",
                heading="Enabling 2FA",
                body=(
                    "You can turn on 2FA from the Security tab under your "
                    "account. Scan the QR code shown there with an "
                    "authenticator app of your choice. After setup, signing "
                    "in will always ask for the six-digit code from the app "
                    "in addition to your password."
                ),
            ),
        ],
        gold_section_ids=[
            "account_security_2fa.md#two-factor-setup",
            "security_faq.md#enabling-2fa",
        ],
    ),
    # ------------------------------------------------------------------ #
    # contradiction — conflicting Sources, curated page picks neither
    # ------------------------------------------------------------------ #
    AdversarialGroup(
        adversarial_class="contradiction",
        group_id="gift-card-expiration",
        wiki_title="Gift Card Expiration",
        wiki_body=(
            "Acme Shop's Source documents disagree on gift card expiration: "
            "gift_card_terms.md states cards never expire, while "
            "gift_card_faq.md states a 12-month expiration with forfeiture "
            "of unused balance. This page intentionally does not pick a "
            "winner — see open_questions."
        ),
        sections=[
            RawSection(
                basename="gift_card_terms.md",
                heading="Expiration",
                body=(
                    "Gift cards issued by Acme Shop never expire and retain "
                    "their full face value indefinitely. Balances carry over "
                    "across account changes and are never forfeited for "
                    "inactivity."
                ),
            ),
            RawSection(
                basename="gift_card_faq.md",
                heading="Expiration",
                body=(
                    "Gift cards expire 12 months after the purchase date. "
                    "Any unused balance remaining after the expiration date "
                    "is forfeited and cannot be reinstated."
                ),
            ),
        ],
        gold_section_ids=[
            "gift_card_terms.md#expiration",
            "gift_card_faq.md#expiration",
        ],
        open_questions=[
            "gift_card_terms.md and gift_card_faq.md directly contradict "
            "each other on gift card expiration (never vs 12 months); "
            "unresolved as of this ingest — do not state either figure as "
            "authoritative until reconciled."
        ],
    ),
    AdversarialGroup(
        adversarial_class="contradiction",
        group_id="free-shipping-threshold",
        wiki_title="Free Shipping Threshold",
        wiki_body=(
            "Acme Shop's Source documents disagree on the free-shipping "
            "threshold: shipping_policy_v2.md states $50, while "
            "promo_terms.md states $75. This page intentionally does not "
            "pick a winner — see open_questions."
        ),
        sections=[
            RawSection(
                basename="shipping_policy_v2.md",
                heading="Free Shipping Threshold",
                body=(
                    "Orders of $50 or more qualify for free standard "
                    "shipping within the continental US. Orders below this "
                    "threshold are charged the standard shipping rate at "
                    "checkout."
                ),
            ),
            RawSection(
                basename="promo_terms.md",
                heading="Free Shipping Threshold",
                body=(
                    "Free shipping applies to orders totaling $75 or more. "
                    "Promotional codes cannot be combined with the "
                    "free-shipping threshold to lower it further."
                ),
            ),
        ],
        gold_section_ids=[
            "shipping_policy_v2.md#free-shipping-threshold",
            "promo_terms.md#free-shipping-threshold",
        ],
        open_questions=[
            "shipping_policy_v2.md and promo_terms.md disagree on the "
            "free-shipping threshold ($50 vs $75); unresolved as of this "
            "ingest."
        ],
    ),
    AdversarialGroup(
        adversarial_class="contradiction",
        group_id="restocking-fee",
        wiki_title="Restocking Fee",
        wiki_body=(
            "Acme Shop's Source documents disagree on whether a restocking "
            "fee applies: returns_policy_addendum.md states none, while "
            "electronics_returns.md states 15% for opened electronics. This "
            "page intentionally does not pick a winner — see "
            "open_questions."
        ),
        sections=[
            RawSection(
                basename="returns_policy_addendum.md",
                heading="Restocking Fee",
                body=(
                    "No restocking fee is charged for standard returns of "
                    "unopened items in their original packaging, regardless "
                    "of category."
                ),
            ),
            RawSection(
                basename="electronics_returns.md",
                heading="Restocking Fee",
                body=(
                    "A 15% restocking fee applies to opened electronics "
                    "returned for any reason other than a manufacturing "
                    "defect."
                ),
            ),
        ],
        gold_section_ids=[
            "returns_policy_addendum.md#restocking-fee",
            "electronics_returns.md#restocking-fee",
        ],
        open_questions=[
            "returns_policy_addendum.md and electronics_returns.md disagree "
            "on whether a restocking fee applies (none vs 15% for opened "
            "electronics); the two Sources may describe non-overlapping "
            "product categories, but neither states that scoping "
            "explicitly — unresolved as of this ingest."
        ],
    ),
    # ------------------------------------------------------------------ #
    # version_evolution — dated Sources, curated page cites only the latest
    # ------------------------------------------------------------------ #
    AdversarialGroup(
        adversarial_class="version_evolution",
        group_id="return-shipping-label-cost",
        wiki_title="Return Shipping Label Cost",
        wiki_body=(
            "Return shipping labels are free for all domestic returns, "
            "regardless of reason (effective 2026-06-01). Two earlier "
            "Source versions (2025-01, 2025-11) described more restrictive "
            "policies now superseded; only the current version is cited "
            "here."
        ),
        sections=[
            RawSection(
                basename="return_shipping_v1.md",
                heading="Label Cost",
                body=(
                    "Customers are responsible for the cost of return "
                    "shipping labels. The label fee is deducted from the "
                    "refund once the item is received."
                ),
                frontmatter={"updated": "2025-01-15"},
            ),
            RawSection(
                basename="return_shipping_v2.md",
                heading="Label Cost",
                body=(
                    "Return shipping labels are provided free of charge "
                    "only when the return reason is a defective or damaged "
                    "item. All other returns still require the customer to "
                    "cover the label cost."
                ),
                frontmatter={
                    "updated": "2025-11-01",
                    "supersedes": "return_shipping_v1.md#label-cost",
                },
            ),
            RawSection(
                basename="return_shipping_v3.md",
                heading="Label Cost",
                body=(
                    "Return shipping labels are now free for all domestic "
                    "returns, regardless of reason. International return "
                    "shipping costs are still the customer's "
                    "responsibility."
                ),
                frontmatter={
                    "updated": "2026-06-01",
                    "supersedes": "return_shipping_v2.md#label-cost",
                },
            ),
        ],
        gold_section_ids=["return_shipping_v3.md#label-cost"],
    ),
    AdversarialGroup(
        adversarial_class="version_evolution",
        group_id="loyalty-gold-tier-threshold",
        wiki_title="Loyalty Gold Tier Threshold",
        wiki_body=(
            "Gold tier status currently requires $1,000 or more in "
            "purchases within a rolling 12-month window (effective "
            "2026-05-15). Two earlier Source versions (2024-01, 2025-06) "
            "described lower thresholds now superseded; only the current "
            "version is cited here."
        ),
        sections=[
            RawSection(
                basename="loyalty_tiers_v1.md",
                heading="Gold Tier Threshold",
                body=(
                    "Gold tier status requires $500 or more in purchases "
                    "within a rolling 12-month window."
                ),
                frontmatter={"updated": "2024-01-10"},
            ),
            RawSection(
                basename="loyalty_tiers_v2.md",
                heading="Gold Tier Threshold",
                body=(
                    "Gold tier status requires $750 or more in purchases "
                    "within a rolling 12-month window, reflecting updated "
                    "reward economics."
                ),
                frontmatter={
                    "updated": "2025-06-01",
                    "supersedes": "loyalty_tiers_v1.md#gold-tier-threshold",
                },
            ),
            RawSection(
                basename="loyalty_tiers_v3.md",
                heading="Gold Tier Threshold",
                body=(
                    "Gold tier status requires $1,000 or more in purchases "
                    "within a rolling 12-month window. Existing Gold "
                    "members are grandfathered at their current spend level "
                    "through the next renewal cycle."
                ),
                frontmatter={
                    "updated": "2026-05-15",
                    "supersedes": "loyalty_tiers_v2.md#gold-tier-threshold",
                },
            ),
        ],
        gold_section_ids=["loyalty_tiers_v3.md#gold-tier-threshold"],
    ),
    AdversarialGroup(
        adversarial_class="version_evolution",
        group_id="warranty-claim-window",
        wiki_title="Warranty Claim Window",
        wiki_body=(
            "Warranty claims must currently be filed within 90 days of "
            "delivery (effective 2026-04-20). Two earlier Source versions "
            "(2024-03, 2025-08) described shorter windows now superseded; "
            "only the current version is cited here."
        ),
        sections=[
            RawSection(
                basename="warranty_claims_v1.md",
                heading="Claim Window",
                body="Warranty claims must be filed within 30 days of delivery.",
                frontmatter={"updated": "2024-03-01"},
            ),
            RawSection(
                basename="warranty_claims_v2.md",
                heading="Claim Window",
                body=(
                    "Warranty claims must be filed within 60 days of "
                    "delivery, an extension from the prior 30-day window."
                ),
                frontmatter={
                    "updated": "2025-08-12",
                    "supersedes": "warranty_claims_v1.md#claim-window",
                },
            ),
            RawSection(
                basename="warranty_claims_v3.md",
                heading="Claim Window",
                body=(
                    "Warranty claims must be filed within 90 days of "
                    "delivery. This is the current claim window; requests "
                    "filed after 90 days are not eligible for warranty "
                    "service."
                ),
                frontmatter={
                    "updated": "2026-04-20",
                    "supersedes": "warranty_claims_v2.md#claim-window",
                },
            ),
        ],
        gold_section_ids=["warranty_claims_v3.md#claim-window"],
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_raw_source(section: RawSection) -> str:
    """Render one raw, docs-native Source file (no `[Source: ...]` citation —
    that convention belongs to synthesized wiki pages, not raw Sources)."""
    body = f"# {section.heading}\n\n{section.body}\n"
    if not section.frontmatter:
        return body
    fm_block = "---\n" + yaml.safe_dump(section.frontmatter, sort_keys=False) + "---\n"
    return f"{fm_block}\n{body}"


def _docs_body_hash(text: str) -> str:
    """Mirrors ``markdown_kb.app.ingest._compute_docs_body_hash`` exactly
    (sha256 of the raw Source file's UTF-8 text) so these fixtures carry
    REAL hashes of their own committed content, not placeholders."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_wiki_concept(group: AdversarialGroup, raw_texts: dict[str, str]) -> str:
    """Render the curated wiki concept page a real ``/ingest`` run would
    produce over `group`'s gold Sections, in the exact frontmatter shape the
    existing corpus v3 wiki fixtures use (``gold.py``'s sole reader
    convention)."""
    gold_sections = [s for s in group.sections if s.source_id in group.gold_section_ids]
    source_hashes = {
        s.basename: {"docs_body": _docs_body_hash(raw_texts[s.basename]), "raw": None}
        for s in gold_sections
    }
    frontmatter = {
        "created": BUILD_TIMESTAMP,
        "id": group.group_id,
        "open_questions": list(group.open_questions),
        "source_hashes": source_hashes,
        "sources": list(group.gold_section_ids),
        "status": "live",
        "type": "concept",
        "updated": BUILD_TIMESTAMP,
    }
    sources_of_truth = ", ".join(s.basename for s in gold_sections)
    sentinel = (
        f"<!-- Auto-generated by POST /ingest on {BUILD_TIMESTAMP}.\n"
        f"     Source of truth: {sources_of_truth}.\n"
        "     Manual edits will be overwritten on next ingest — edit the "
        "Source instead. -->"
    )
    fm_block = (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---"
    )
    citations = "\n".join(f"[Source: {sid}]" for sid in group.gold_section_ids)
    return (
        f"{sentinel}\n\n{fm_block}\n\n# {group.wiki_title}\n\n{group.wiki_body}\n\n"
        f"{citations}\n"
    )


# ---------------------------------------------------------------------------
# Public API — regeneration entry point
# ---------------------------------------------------------------------------
def write_corpus_fixtures(ledger: CostLedger | None = None) -> CostLedger:
    """(Re)write every raw Source and curated wiki concept page in
    ``ADVERSARIAL_GROUPS`` under this package's own ``corpus/`` and
    ``wiki/concepts/`` dirs ONLY — never touches the repo's production
    ``docs/`` or ``wiki/`` (production isolation, PRD #654). Idempotent:
    re-running reproduces byte-identical output (fixed timestamp, real
    content hashes, no wall-clock or random input).

    Records one zero-usage ``CostLedger`` "build" entry per group (see
    module docstring) and returns the ledger so a caller (the CLI below, or
    a test) can inspect or persist it.
    """
    ledger = ledger if ledger is not None else CostLedger()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)

    for group in ADVERSARIAL_GROUPS:
        raw_texts: dict[str, str] = {}
        for section in group.sections:
            text = render_raw_source(section)
            raw_texts[section.basename] = text
            (CORPUS_DIR / section.basename).write_text(text, encoding="utf-8")
        wiki_text = render_wiki_concept(group, raw_texts)
        (WIKI_CONCEPTS_DIR / f"{group.group_id}.md").write_text(
            wiki_text, encoding="utf-8"
        )
        ledger.record(
            stack="wiki_curation",
            phase="build",
            model="offline-deterministic",
            usage=UsageMetadata(),
        )
    return ledger


# §6.6: a run that produces anything less than real data (here: no LLM
# synthesis call was made — construction was offline/hand-authored, see
# module docstring) writes to a trust-marked path, never the canonical name,
# led by this loud header (§3.3 sentinel-string constant).
BUILD_COST_REPORT_PLACEHOLDER_HEADER = (
    "⚠️ PLACEHOLDER — NOT A LIVE-SYNTHESIS COST MEASUREMENT. This ledger "
    "reflects the offline, deterministic, hand-authored construction method "
    "(no LLM call was made to build this corpus); its zero is real for THAT "
    'method but must not be read as "the wiki corpus build costs $0" in '
    "general — ADR-0045 cites a real live-synthesis build cost "
    "(~$4.4/corpus). Re-run via a live corpus-build script for the real figure."
)
BUILD_COST_REPORT_PATH = _PKG_ROOT / "BUILD_COST.offline-tracer.md"


def write_build_cost_report(ledger: CostLedger, path: Path) -> None:
    """Persist `ledger`'s totals as a short, trust-marked Markdown report
    (PRD #654 user story 18: "build cost ... recorded per stack via the
    cost ledger during construction"; CODING_STANDARD §6.6: an offline run
    writes to a `*.offline-tracer.*` path with a loud placeholder header,
    never the canonical name)."""
    totals = ledger.totals(stack="wiki_curation", phase="build")
    lines = [
        BUILD_COST_REPORT_PLACEHOLDER_HEADER,
        "",
        "# Corpus v3 adversarial fixtures — build cost ledger",
        "",
        "> Generated by `eval/corpus_v3/build_corpus.py`. Construction method: "
        "offline, deterministic, hand-authored (see that module's docstring).",
        "",
        "| Stack | Phase | Calls | Input tokens | Output tokens | Total tokens | USD |",
        "|---|---|---|---|---|---|---|",
        f"| {totals.stack} | {totals.phase} | {totals.calls} | {totals.input_tokens} "
        f"| {totals.output_tokens} | {totals.total_tokens} | "
        f"{'unpriced (offline-deterministic model)' if totals.usd is None else totals.usd} |",
        "",
        f"{len(ADVERSARIAL_GROUPS)} adversarial groups built "
        f"({sum(1 for g in ADVERSARIAL_GROUPS if g.adversarial_class == 'redundancy')} "
        "redundancy, "
        f"{sum(1 for g in ADVERSARIAL_GROUPS if g.adversarial_class == 'contradiction')} "
        "contradiction, "
        f"{sum(1 for g in ADVERSARIAL_GROUPS if g.adversarial_class == 'version_evolution')} "
        "version_evolution), one ledger entry per group.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    built_ledger = write_corpus_fixtures()
    write_build_cost_report(built_ledger, BUILD_COST_REPORT_PATH)
    print(
        f"Wrote {len(ADVERSARIAL_GROUPS)} adversarial groups to {CORPUS_DIR} / "
        f"{WIKI_CONCEPTS_DIR}; build cost report at {BUILD_COST_REPORT_PATH}"
    )
