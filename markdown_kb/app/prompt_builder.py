"""Shallow module per Ousterhout. Public surface: ``SYSTEM_PROMPT``, ``build_prompt``.

Prompt builder for the grounded Q&A assistant.

Owns the SYSTEM_PROMPT (5 numbered rules per the PROMPT.md spec) and
build_prompt(), which assembles the CONTEXT: / QUESTION: block that is
passed to the LLM as a HumanMessage.

Prompt structure (per PROMPT.md Q3):
    CONTEXT:
    [Source: filename#heading]
    Heading: parent > leaf
    <section body>

    QUESTION:
    <user question>
"""

from __future__ import annotations

from .indexer import Section

# ---------------------------------------------------------------------------
# System prompt — 5 numbered grounding rules (strict per ADR-0001)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict knowledge-base assistant. Follow these rules exactly:

1. Answer ONLY using the information in the CONTEXT section below. Do not use outside world knowledge, training data, or inference beyond what is written.
2. Every factual claim in your answer MUST cite at least one source using the exact format: [Source: filename#heading]. Use the Citation ids as they appear in the CONTEXT headers.
3. If the CONTEXT does not contain enough information to answer the question, reply with the exact phrase: "I cannot confirm from the knowledge base." — nothing more, nothing less.
4. You may synthesize information across multiple cited Sections if needed, but every claim must still trace to a cited Section.
5. Never guess, never infer beyond the text, never complete gaps with general knowledge. "I cannot confirm from the knowledge base." is a good, expected answer — not a failure.
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(question: str, ranked_sections: list[Section]) -> str:
    """Build the [Human] message for the LLM from ranked Sections.

    Structure:
        CONTEXT:

        [Source: filename#heading]
        Heading: parent > leaf
        <section content>

        (repeated for each section)

        QUESTION:
        <question>

    Scores are NOT included in the prompt (per PROMPT.md Q3: prevents
    the model reasoning "low score → guess").
    """
    parts: list[str] = ["CONTEXT:\n"]

    for sec in ranked_sections:
        breadcrumb = " > ".join(sec.heading_path)
        block = f"[Source: {sec.id}]\nHeading: {breadcrumb}\n{sec.content}\n"
        parts.append(block)

    parts.append(f"\nQUESTION:\n{question}")
    return "\n".join(parts)
