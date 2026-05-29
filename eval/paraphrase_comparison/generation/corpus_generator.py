"""Acme-Shop synthetic corpus generator (Phase 8.5 S5, issue #143).

Public surface: ``DocSpec``, ``SectionSpec``, ``DOC_SPECS``, ``generate_scaffold``,
``generate_doc_live``, ``CORPUS_ENTITY_SOURCES``.

This module is the **documented way a user regenerates or extends the
``docs/fake-docs/`` corpus**.  It defines the full Acme-Shop document plan as a
static Python data structure (``DOC_SPECS``) so the layout, headings, and
frontmatter shape are **deterministic and offline-testable** without an API key.

Two generation modes
--------------------
Offline scaffold
    ``generate_scaffold(doc, output_dir)`` writes a syntactically valid Markdown
    Source with minimal stub content derived from the heading text.  No LLM call.
    Useful for structural tests and for bootstrapping a new doc before the live
    run.  Offline tests assert on scaffold output only.

Live generation (opt-in, ``@pytest.mark.live``)
    ``generate_doc_live(doc, output_dir, llm)`` replaces each section body with
    realistic prose produced by the supplied OpenAI LLM client.  This is the
    deliverable for issue #145 (Demo-tier live regeneration + trust review, HITL).

CLI (from repo root)
--------------------
    # Scaffold only (no API key needed):
    uv run python -m eval.paraphrase_comparison.generation.corpus_generator

    # Live regeneration (requires OPENAI_API_KEY, see issue #145):
    uv run python -m eval.paraphrase_comparison.generation.corpus_generator --live

    # Regenerate a single doc (scaffold):
    uv run python -m eval.paraphrase_comparison.generation.corpus_generator \\
        --doc returns_policy

Global-uniqueness constraint
----------------------------
Basenames in ``docs/`` must be globally unique (ADR requirement, documented in
``docs/README.md``).  ``DOC_SPECS`` encodes basenames as explicit fields; the
unit tests enforce no collisions within the plan and no collisions against the
pre-existing ``docs/`` tree.

Stdout via ``print`` is acceptable here because this is a one-off CLI script,
not a committed library (CODING_STANDARD §5.1).
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
FAKE_DOCS_DIR = _REPO_ROOT / "docs" / "fake-docs"

# Entity sources are excluded from the Gold Section pool (they collapse into a
# single entity wiki page).  This set mirrors CORPUS_ENTITY_SOURCES in
# sampling.py and is the one piece of hand-knowledge in the generator.
CORPUS_ENTITY_SOURCES: frozenset[str] = frozenset({"warranty.md", "acme_shop_about.md"})

# Prefix written at the top of every scaffold doc so tools can identify it.
_SCAFFOLD_SENTINEL = "<!-- scaffold: generated stub — replace with live content -->"

# Generator model for the live path (same as generate_paraphrases.py convention).
_GENERATOR_MODEL = "gpt-4o"
_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionSpec:
    """Specification for one ## section within a Source document.

    ``heading`` is the exact heading text (no ``##`` prefix).
    ``prompt_hint`` is a one-sentence description of the section's content,
    used as the instruction in the live-generation prompt.
    """

    heading: str
    prompt_hint: str


@dataclass(frozen=True)
class DocSpec:
    """Specification for one Acme-Shop Source document.

    ``basename`` — the filename under ``docs/fake-docs/`` (must be globally
    unique across all of ``docs/``).
    ``title`` — the H1 heading (no ``#`` prefix).
    ``sections`` — ordered list of ``SectionSpec`` instances.
    ``is_entity`` — when True the doc is an entity Source (excluded from the
    Gold Section pool; e.g. the about/warranty pages).
    """

    basename: str
    title: str
    sections: tuple[SectionSpec, ...]
    is_entity: bool = False

    @property
    def gold_section_count(self) -> int:
        """Number of Gold-eligible sections this doc contributes."""
        return 0 if self.is_entity else len(self.sections)


# ---------------------------------------------------------------------------
# Document plan — the canonical Acme-Shop corpus definition
# ---------------------------------------------------------------------------

DOC_SPECS: tuple[DocSpec, ...] = (
    # -----------------------------------------------------------------------
    # Entity sources (excluded from Gold Section pool)
    # -----------------------------------------------------------------------
    DocSpec(
        basename="acme_shop_about.md",
        title="Acme Shop",
        is_entity=True,
        sections=(
            SectionSpec(
                "Company History", "Brief history of Acme Shop's founding and growth."
            ),
            SectionSpec(
                "Mission", "Acme Shop's mission statement and core principles."
            ),
            SectionSpec(
                "Team Summary",
                "Key members of the Acme Shop leadership and founding team.",
            ),
            SectionSpec(
                "Locations", "Where Acme Shop warehouses and offices are located."
            ),
            SectionSpec("Contact", "How customers and press can contact Acme Shop."),
        ),
    ),
    DocSpec(
        basename="warranty.md",
        title="Warranty",
        is_entity=True,
        sections=(
            SectionSpec(
                "Coverage Period",
                "What is covered and for how long under the standard warranty.",
            ),
            SectionSpec(
                "Warranty Claim", "Steps a customer follows to file a warranty claim."
            ),
        ),
    ),
    # -----------------------------------------------------------------------
    # Concept sources (Gold Section pool)
    # -----------------------------------------------------------------------
    DocSpec(
        basename="account_management.md",
        title="Account Management",
        sections=(
            SectionSpec("Password Reset", "How to reset a forgotten account password."),
            SectionSpec(
                "Update Email Address", "How to change the email address on an account."
            ),
            SectionSpec("Close Account", "How to permanently close an account."),
        ),
    ),
    DocSpec(
        basename="account_security.md",
        title="Account Security",
        sections=(
            SectionSpec(
                "Two-Factor Authentication",
                "Setting up and using two-factor authentication.",
            ),
            SectionSpec(
                "Suspicious Activity",
                "What happens when suspicious login activity is detected.",
            ),
            SectionSpec(
                "Managing Trusted Devices", "How to view and remove trusted devices."
            ),
        ),
    ),
    DocSpec(
        basename="bulk_orders.md",
        title="Bulk Orders",
        sections=(
            SectionSpec(
                "Eligibility and Minimum Order",
                "Who qualifies for bulk orders and the minimum quantity required.",
            ),
            SectionSpec(
                "Volume Discounts", "Discount tiers based on quantity ordered."
            ),
            SectionSpec(
                "Fulfilment and Lead Times",
                "How bulk orders are shipped and expected lead times.",
            ),
        ),
    ),
    DocSpec(
        basename="customer_support.md",
        title="Customer Support",
        sections=(
            SectionSpec(
                "Contact Channels",
                "Ways to reach the support team: chat, email, and phone.",
            ),
            SectionSpec(
                "Escalation Process",
                "How unresolved issues are escalated to specialist teams.",
            ),
            SectionSpec(
                "Self-Service Resources",
                "The help centre, FAQs, and virtual assistant.",
            ),
        ),
    ),
    DocSpec(
        basename="damaged_items.md",
        title="Damaged Items",
        sections=(
            SectionSpec(
                "Reporting Damage",
                "How to report a damaged delivery within the reporting window.",
            ),
            SectionSpec(
                "Replacement Process", "What happens after a damage report is approved."
            ),
        ),
    ),
    DocSpec(
        basename="gift_cards.md",
        title="Gift Cards",
        sections=(
            SectionSpec(
                "Purchase and Delivery",
                "How to buy digital and physical gift cards and when they are delivered.",
            ),
            SectionSpec(
                "Redeeming a Gift Card", "How to apply a gift card balance at checkout."
            ),
            SectionSpec("Lost Gift Cards", "Policy for replacing a lost gift card."),
        ),
    ),
    DocSpec(
        basename="international_shipping.md",
        title="International Shipping",
        sections=(
            SectionSpec(
                "Supported Countries",
                "Which countries Acme Shop ships to and how to check.",
            ),
            SectionSpec(
                "Customs and Duties",
                "How import duties and taxes are handled for international orders.",
            ),
            SectionSpec(
                "International Delivery Times",
                "Expected transit times for international shipments.",
            ),
        ),
    ),
    DocSpec(
        basename="loyalty_program.md",
        title="Loyalty Program",
        sections=(
            SectionSpec("Earning Points", "How customers earn Acme Rewards points."),
            SectionSpec(
                "Redeeming Points", "How to redeem points for account credits."
            ),
            SectionSpec(
                "Membership Tiers", "The three loyalty tiers and their benefits."
            ),
        ),
    ),
    DocSpec(
        basename="order_management.md",
        title="Order Management",
        sections=(
            SectionSpec("Cancel an Order", "How and when an order can be cancelled."),
            SectionSpec(
                "Change Shipping Address",
                "How to update a shipping address before the order ships.",
            ),
            SectionSpec(
                "Combine Orders", "How to merge two separate orders into one shipment."
            ),
            SectionSpec(
                "Backordered Items",
                "What happens when an item in an order is backordered.",
            ),
        ),
    ),
    DocSpec(
        basename="payment_methods.md",
        title="Payment Methods",
        sections=(
            SectionSpec("Accepted Cards", "Which payment methods Acme Shop accepts."),
            SectionSpec(
                "Payment Authorization", "How payment is authorised and captured."
            ),
            SectionSpec(
                "Failed Payments",
                "What happens when a payment is declined and how to resolve it.",
            ),
        ),
    ),
    DocSpec(
        basename="price_matching.md",
        title="Price Matching",
        sections=(
            SectionSpec(
                "Price Match Eligibility",
                "Conditions under which Acme Shop will match a competitor's price.",
            ),
            SectionSpec(
                "Requesting a Price Match",
                "How to submit a price-match request and the time limit.",
            ),
        ),
    ),
    DocSpec(
        basename="product_care.md",
        title="Product Care",
        sections=(
            SectionSpec(
                "Cleaning Instructions", "How to clean Acme Shop products safely."
            ),
            SectionSpec(
                "Storage Guidelines",
                "Proper storage conditions to extend product life.",
            ),
            SectionSpec(
                "Warranty Implications", "How improper care affects warranty coverage."
            ),
        ),
    ),
    DocSpec(
        basename="product_information.md",
        title="Product Information",
        sections=(
            SectionSpec(
                "Stock Availability",
                "How to check whether a product is in stock and set up restock alerts.",
            ),
            SectionSpec(
                "Product Reviews", "How customer reviews are collected and displayed."
            ),
            SectionSpec(
                "Size Guide",
                "How to use Acme Shop's sizing charts to find the right fit.",
            ),
        ),
    ),
    DocSpec(
        basename="promo_codes.md",
        title="Promo Codes",
        sections=(
            SectionSpec(
                "Applying a Code", "How to apply a promotional code at checkout."
            ),
            SectionSpec(
                "Stacking Rules",
                "Which combinations of promotional codes are permitted.",
            ),
        ),
    ),
    DocSpec(
        basename="returns_policy.md",
        title="Returns Policy",
        sections=(
            SectionSpec(
                "Return Window",
                "How long customers have to return an item and the conditions required.",
            ),
            SectionSpec(
                "Refund Processing Time",
                "How long refunds take after a return is received.",
            ),
            SectionSpec(
                "Restocking Fee",
                "Which items carry a restocking fee and the percentage charged.",
            ),
        ),
    ),
    DocSpec(
        basename="shipping_options.md",
        title="Shipping Options",
        sections=(
            SectionSpec("Standard Delivery", "Standard delivery timelines and costs."),
            SectionSpec(
                "Expedited Delivery", "Expedited and next-day delivery options."
            ),
            SectionSpec("Order Tracking", "How customers track their shipments."),
        ),
    ),
    DocSpec(
        basename="store_pickup.md",
        title="Store Pickup",
        sections=(
            SectionSpec(
                "Buy Online Pick Up In Store",
                "How to place an order for in-store pickup.",
            ),
            SectionSpec(
                "Pickup Identification",
                "What identification is needed to collect an order.",
            ),
        ),
    ),
    DocSpec(
        basename="subscription_orders.md",
        title="Subscription Orders",
        sections=(
            SectionSpec(
                "Setting Up a Subscription",
                "How to enrol a product in auto-replenishment.",
            ),
            SectionSpec(
                "Managing a Subscription",
                "How to pause, skip, or cancel a subscription.",
            ),
            SectionSpec(
                "Subscription Pricing Changes",
                "How price changes affect active subscriptions.",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(heading: str) -> str:
    """Return a URL-safe slug from a heading string (mirrors markdown_kb.slugify)."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _scaffold_body(section: SectionSpec) -> str:
    """Return a deterministic stub body for a SectionSpec.

    The stub is derived entirely from the heading and prompt_hint so it is
    offline-reproducible without any LLM call.  It is intentionally minimal and
    clearly marked as synthetic scaffolding so reviewers know it is a placeholder.
    """
    wrapped_hint = textwrap.fill(section.prompt_hint, width=72)
    return (
        f"[SCAFFOLD: {section.heading}] "
        f"{wrapped_hint} "
        f"This section will be replaced with realistic prose during the live "
        f"generation run (issue #145)."
    )


# ---------------------------------------------------------------------------
# Offline scaffold generation
# ---------------------------------------------------------------------------


def generate_scaffold(doc: DocSpec, output_dir: Path) -> Path:
    """Write a syntactically valid Markdown stub for *doc* into *output_dir*.

    The output has the correct H1 + H2 structure and valid frontmatter shape so
    ``derive_gold_sections`` can parse it.  Section bodies are deterministic stub
    strings (no LLM call).  The file is always (re-)written so the function is
    idempotent.

    Returns the path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / doc.basename

    lines: list[str] = [_SCAFFOLD_SENTINEL, "", f"# {doc.title}", ""]
    for section in doc.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        lines.append(_scaffold_body(section))
        lines.append("")

    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Live generation (opt-in — requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def _get_live_llm():
    """Return a gpt-4o ChatOpenAI client (lazy import — keeps LangChain internal)."""
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 — function-scope per ADR-0005

    return ChatOpenAI(
        model=_GENERATOR_MODEL,
        temperature=_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


def generate_doc_live(doc: DocSpec, output_dir: Path, llm) -> Path:
    """Write a live-generated Markdown Source for *doc* into *output_dir*.

    Calls the supplied OpenAI LLM client once per section to produce realistic
    e-commerce help-desk prose.  The resulting file has the same H1 + H2
    structure as the scaffold but with LLM-written bodies instead of stubs.

    This is the opt-in live path — only called when ``OPENAI_API_KEY`` is set
    and the ``--live`` flag is passed.  The full live run is tracked as issue
    #145 (Demo-tier live regeneration + trust review, HITL).

    Returns the path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / doc.basename

    lines: list[str] = [f"# {doc.title}", ""]
    for section in doc.sections:
        prompt = (
            f"Write a concise, realistic help-centre section for an e-commerce "
            f"company called Acme Shop. The section heading is "
            f'"{section.heading}" under the page "{doc.title}". '
            f"Content focus: {section.prompt_hint} "
            f"Write 3–5 sentences of factual, professional prose in plain English. "
            f"Do not include the heading itself in your response."
        )
        response = llm.invoke(prompt)
        body = response.content.strip()
        lines.append(f"## {section.heading}")
        lines.append("")
        lines.append(body)
        lines.append("")

    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for the corpus generator CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Acme-Shop synthetic Source pool into docs/fake-docs/. "
            "Without --live, writes deterministic scaffold stubs (no API key needed). "
            "With --live (requires OPENAI_API_KEY), writes realistic LLM prose. "
            "Full live regeneration is tracked as issue #145."
        )
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run the live generation path (requires OPENAI_API_KEY). "
            "Writes LLM-generated prose instead of scaffold stubs. "
            "Issue #145: Demo-tier live regeneration + trust review (HITL)."
        ),
    )
    parser.add_argument(
        "--doc",
        metavar="BASENAME_OR_TITLE",
        help=(
            "Regenerate only the named doc (basename without .md, or title keyword). "
            "If omitted, all docs in DOC_SPECS are processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(FAKE_DOCS_DIR),
        help=f"Output directory (default: {FAKE_DOCS_DIR}).",
    )
    args = parser.parse_args(argv)
    load_dotenv(
        find_dotenv(usecwd=True)
    )  # pick up OPENAI_API_KEY from a repo-root .env

    output_dir = Path(args.output_dir)

    # Filter to a single doc if requested.
    targets: tuple[DocSpec, ...] = DOC_SPECS
    if args.doc:
        query = args.doc.lower().removesuffix(".md")
        targets = tuple(
            d
            for d in DOC_SPECS
            if d.basename.removesuffix(".md") == query or query in d.title.lower()
        )
        if not targets:
            print(f"ERROR: no doc matching {args.doc!r}. Available basenames:")
            for d in DOC_SPECS:
                print(f"  {d.basename}")
            return 1

    if args.live:
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "ERROR: --live requires OPENAI_API_KEY to be set. "
                "For the full live regeneration workflow, see issue #145."
            )
            return 1
        llm = _get_live_llm()
        for doc in targets:
            dest = generate_doc_live(doc, output_dir, llm)
            gold_count = doc.gold_section_count
            print(
                f"  LIVE  {doc.basename} ({len(doc.sections)} sections, "
                f"{gold_count} Gold-eligible) → {dest}"
            )
    else:
        for doc in targets:
            dest = generate_scaffold(doc, output_dir)
            gold_count = doc.gold_section_count
            print(
                f"  SCAFFOLD  {doc.basename} ({len(doc.sections)} sections, "
                f"{gold_count} Gold-eligible) → {dest}"
            )

    total_gold = sum(d.gold_section_count for d in targets)
    print(
        f"\nDone: {len(targets)} doc(s) written, {total_gold} Gold Section(s) across targets."
    )
    if not args.live:
        print(
            "\nNote: scaffold mode writes stub content only. "
            "Run with --live (OPENAI_API_KEY required) for realistic prose. "
            "See issue #145 for the full Demo-tier live regeneration workflow."
        )

    # Verify derive_gold_sections picks up the generated docs.
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections  # noqa: PLC0415

    derived = derive_gold_sections(output_dir)
    print(
        f"\nderive_gold_sections sees {len(derived)} Gold Section(s) in {output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
