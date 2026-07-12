"""Tests for the qa question-token downweight (issue #578, rule 2a follow-up).

Rule 2a (#570) joins a qa page's frontmatter ``question:`` value into its BM25
tokens so a filed answer is retrievable BY its own question. At scale that
creates a new collision: two qa pages that share only a generic interrogative
("你們"/"哪些" as CJK bigrams, "how"/"what" as English words) can compete on
those shared tokens alone, letting a qa page that is NOT about the query's
real topic out-rank the Section that actually carries the answer (observed
live, issue #578: a payment qa page ranking over the shipping-countries fact
for a "you ship to which countries" style query).

``bm25_score``'s ``qa_question_weight`` parameter (default
``indexer.QA_QUESTION_TOKEN_WEIGHT``, calibrated by
``eval/qa_field_weight/calibrate.py``) scales the term-frequency contribution
of token occurrences that came ONLY from the injected question, never from
real body content. These tests exercise ``bm25_score``/``Section`` directly
(no file I/O) per the existing ``test_indexer_qa_question_tokens.py``
convention of testing external behaviour, but at the scoring-function level
since the property under test is about relative SCORE contribution, not
about what tokens end up in the index.
"""

from __future__ import annotations

import app.indexer as indexer_module
from app.indexer import Section


def _snapshot_module_state():
    return (
        list(indexer_module.sections),
        indexer_module.doc_freq.copy(),
        indexer_module.avg_doc_len,
        indexer_module.files_indexed,
    )


def _restore_module_state(snapshot) -> None:
    sections, doc_freq, avg_doc_len, files_indexed = snapshot
    indexer_module.sections = sections
    indexer_module.doc_freq = doc_freq
    indexer_module.avg_doc_len = avg_doc_len
    indexer_module.files_indexed = files_indexed


def _install_corpus(noise: Section, real: Section) -> None:
    """Replace the module-level index with exactly these two Sections.

    Mirrors ``build_index``'s bookkeeping (``rebuild_stats``) without any file
    I/O, since the property under test lives entirely in ``bm25_score``.
    """
    indexer_module.sections = [noise, real]
    indexer_module.rebuild_stats()


def test_default_weight_is_the_calibrated_value():
    """The shipped default is the ``eval/qa_field_weight/calibrate.py`` recommendation.

    Pins the constant to its calibration evidence (CODING_STANDARD §4.3 / #253
    precedent) so a future hand-edit of the default without re-running the
    sweep is caught here rather than silently drifting.
    """
    assert indexer_module._QA_QUESTION_TOKEN_WEIGHT_DEFAULT == 0.3
    assert 0.0 < indexer_module._QA_QUESTION_TOKEN_WEIGHT_DEFAULT < 1.0, (
        "the scope decision is 'downweight', not remove (0.0) or no-op (1.0)"
    )


def test_qa_pollution_scenario_before_after_weight(monkeypatch):
    """Reproduces the #578 symptom: a downweight fixes ranking, weight=1.0 doesn't.

    ``noise`` is a qa page about payment (question shares the generic "you"/
    "which" interrogative with the query but its body never mentions
    shipping/countries). ``real`` is a plain concept Section that legitimately
    carries the shipping-countries fact but has no interrogative overlap with
    the query at all -- exactly the #578 shape.
    """
    snapshot = _snapshot_module_state()
    monkeypatch.setattr(indexer_module, "sections", indexer_module.sections, raising=False)

    noise_question_tokens = ["you", "which", "payment", "methods"]
    noise = Section(
        id="noise-qa#noise-qa",
        file="noise-qa",
        heading="noise-qa",
        heading_path=["noise-qa"],
        content="Visa and Mastercard are accepted.",
        tokens=noise_question_tokens + ["visa", "mastercard", "accepted"],
        metadata={"type": "qa", "question": "you which payment methods"},
        question_tokens=noise_question_tokens,
    )
    real = Section(
        id="shipping#countries",
        file="shipping",
        heading="countries",
        heading_path=["shipping", "countries"],
        content="Ships to Japan, Korea, and Canada.",
        tokens=["ships", "japan", "korea", "canada", "countries"],
        metadata={"type": "concept"},
    )
    _install_corpus(noise, real)
    try:
        query_tokens = ["you", "which", "countries"]

        # BEFORE (weight=1.0, the pre-#578 behaviour): the noise qa page's
        # score is inflated by two full-weight matches ("you", "which") it
        # only carries via the injected question, out-ranking (or matching)
        # the real content that answers the query via one weaker match
        # ("countries").
        noise_at_1 = indexer_module.bm25_score(query_tokens, noise, qa_question_weight=1.0)
        real_at_1 = indexer_module.bm25_score(query_tokens, real, qa_question_weight=1.0)
        assert noise_at_1 > real_at_1, (
            "setup sanity check: at weight=1.0 the noise qa page must "
            f"out-rank the real content (noise={noise_at_1:.3f}, "
            f"real={real_at_1:.3f}) -- otherwise this fixture doesn't "
            "reproduce the #578 symptom"
        )

        # AFTER (calibrated default weight < 1.0): the same two question-only
        # matches are downweighted, and the real content now out-ranks the
        # noise qa page for the query it should never have won.
        noise_at_default = indexer_module.bm25_score(query_tokens, noise)
        real_at_default = indexer_module.bm25_score(query_tokens, real)
        assert real_at_default > noise_at_default, (
            "at the calibrated default weight, real content matched on its "
            f"own topic must out-rank a qa page matched only on a shared "
            f"interrogative (noise={noise_at_default:.3f}, "
            f"real={real_at_default:.3f})"
        )

        # The downweight strictly lowers the noise page's score relative to
        # the pre-#578 baseline -- it does not touch the real content's score
        # at all (it carries no question_tokens).
        assert noise_at_default < noise_at_1
        assert real_at_default == real_at_1
    finally:
        _restore_module_state(snapshot)


def test_non_qa_section_score_unaffected_by_weight(monkeypatch):
    """A Section with no ``question_tokens`` scores identically at any weight.

    ``question_tokens`` defaults to ``[]`` for every Section built before
    #578 existed (rule 2 pages, and any qa page with no ``question``), so the
    downweight must be a complete no-op for them.
    """
    snapshot = _snapshot_module_state()
    sec = Section(
        id="concept#a",
        file="concept",
        heading="a",
        heading_path=["concept", "a"],
        content="Returns are accepted within 30 days.",
        tokens=["returns", "accepted", "within", "30", "days"],
        metadata={"type": "concept"},
    )
    _install_corpus(sec, sec)
    try:
        query_tokens = ["returns", "days"]
        score_weight_0 = indexer_module.bm25_score(query_tokens, sec, qa_question_weight=0.0)
        score_weight_1 = indexer_module.bm25_score(query_tokens, sec, qa_question_weight=1.0)
        score_default = indexer_module.bm25_score(query_tokens, sec)
        assert score_weight_0 == score_weight_1 == score_default
    finally:
        _restore_module_state(snapshot)


def test_own_question_still_retrievable_at_default_weight(monkeypatch):
    """Rule 2a's own-question retrievability survives the default downweight.

    Regression guard for #570: a qa page whose body shares NO distinctive
    token with its own question must still rank for that question at the
    calibrated default weight (weight=0 would fully undo #570; the scope
    decision is "downweight", not "remove").
    """
    snapshot = _snapshot_module_state()
    question_tokens = ["countries", "ship"]
    qa_page = Section(
        id="which-countries#which-countries",
        file="which-countries",
        heading="which-countries",
        heading_path=["which-countries"],
        content="Delivered worldwide, allegedly.",
        tokens=question_tokens + ["delivered", "worldwide", "allegedly"],
        metadata={"type": "qa", "question": "Which countries do you ship to?"},
        question_tokens=question_tokens,
    )
    unrelated = Section(
        id="returns#returns",
        file="returns",
        heading="returns",
        heading_path=["returns"],
        content="Returns are accepted within 30 days.",
        tokens=["returns", "accepted", "within", "30", "days"],
        metadata={"type": "concept"},
    )
    _install_corpus(qa_page, unrelated)
    try:
        score = indexer_module.bm25_score(["countries", "ship"], qa_page)
        assert score > 0, (
            "a qa page must still score above zero for its own question at "
            "the default downweight -- rule 2a must not be fully undone"
        )
    finally:
        _restore_module_state(snapshot)
