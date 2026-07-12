"""QA question-field token weight calibration (issue #578, rule 2a follow-up).

Rule 2a (#570) joins a filed qa page's frontmatter ``question:`` into its BM25
tokens so it is retrievable by its own question. At scale that creates a new
collision: qa pages that share only a generic interrogative (CJK "你們"/
"哪些" bigrams are never stopword-filtered — see ``indexer.STOP_WORDS``) can
out-rank the Section that actually carries the answer. This package sweeps
``markdown_kb.app.indexer.QA_QUESTION_TOKEN_WEIGHT`` (which downweights a qa
Section's question-only token matches relative to its body matches) against a
synthetic corpus that reproduces the collision AT SCALE, and against the real
rule-2a own-question invariant (#570) that a downweight must not break — so the
shipped default is chosen by evidence (CODING_STANDARD §4.3), not hand-picked.
LLM-free and deterministic, mirroring ``eval/negative_case/calibrate.py``.
"""
