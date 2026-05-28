"""Deep module per Ousterhout. Public surface: ``build_corpus_idf``, ``rekey_paraphrase``, ``rekey_queries``.

Deterministic re-keying of the committed ``queries.yaml`` Key Tokens (issue #139).

Replaces the LLM-emitted (often multi-word, unmatchable) Key Tokens with tokens
derived deterministically from each Paraphrase's Gold Section body via corpus IDF.
After re-keying, ``key_tokens_docs`` and ``key_tokens_wiki`` are identical (the
dual-side collapse the issue calls for) — both carry the same IDF-ranked single
tokens from the docs Gold Section body.

CLI usage (from repo root)::

    uv run python -m eval.paraphrase_comparison.rekey

The script overwrites ``queries.yaml`` atomically (CODING_STANDARD §2.6) and
prints a brief summary. It can also be called programmatically in tests via
``rekey_queries``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from markdown_kb.app.indexer import parse_markdown, slugify

from .generation.qc import build_idf, derive_key_tokens
from .generate_paraphrases import render_queries_yaml
from .loader import QUERIES_PATH, load_paraphrases, write_text_atomic
from .models import Paraphrase

_PKG_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = _PKG_ROOT / "corpus"

# Top-N distinctive tokens per Gold Section body (issue #139).
# Tuned to cover a meaningful portion of the section's content signal while
# staying small enough that every token is genuinely distinctive.
DEFAULT_TOP_N = 10


# ---------------------------------------------------------------------------
# Corpus IDF builder
# ---------------------------------------------------------------------------
def build_corpus_idf(corpus_dir: Path = CORPUS_DIR) -> dict[str, float]:
    """Build a corpus-wide IDF table from all docs Gold Section bodies.

    Tokenises every Section in every ``*.md`` file under ``corpus_dir`` using
    the markdown_kb tokeniser (ADR-0002) so stop-words and punctuation are
    stripped consistently with the BM25 index and the C5c metric.
    """
    bodies: list[str] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        for section in parse_markdown(md_file, source_id=None):
            if section.content.strip():
                bodies.append(section.content)
    return build_idf(bodies)


# ---------------------------------------------------------------------------
# Per-Paraphrase re-keying
# ---------------------------------------------------------------------------
def rekey_paraphrase(
    paraphrase: Paraphrase,
    section_body: str,
    idf: dict[str, float],
    top_n: int = DEFAULT_TOP_N,
) -> Paraphrase:
    """Return a new ``Paraphrase`` with deterministic IDF Key Tokens.

    Both ``key_tokens_docs`` and ``key_tokens_wiki`` are set to the same
    IDF-ranked token list derived from ``section_body`` (dual-side collapse per
    issue #139). The ``text``, ``gold_docs_section_id``, and all other fields
    are unchanged.
    """
    tokens = derive_key_tokens(section_body, idf, top_n=top_n)
    return replace(paraphrase, key_tokens_docs=tokens, key_tokens_wiki=tokens)


# ---------------------------------------------------------------------------
# Full re-keying pass
# ---------------------------------------------------------------------------
def rekey_queries(
    queries_path: Path = QUERIES_PATH,
    corpus_dir: Path = CORPUS_DIR,
    top_n: int = DEFAULT_TOP_N,
) -> list[Paraphrase]:
    """Re-key every Paraphrase in ``queries_path`` and overwrite the file atomically.

    Steps:
      1. Build corpus IDF from ``corpus_dir``.
      2. Load Gold Section bodies indexed by ``{file}#{slug}``.
      3. For each Paraphrase, derive deterministic Key Tokens from its Gold
         Section body and replace both dual-side lists.
      4. Overwrite ``queries_path`` atomically with the updated YAML.
      5. Return the rekeyed Paraphrase list (for callers / tests).

    Paraphrases whose Gold Section id is not found in the corpus are kept with
    their original Key Tokens and a warning is printed — this signals a corpus
    drift that a human should investigate, not an automatic drop.
    """
    idf = build_corpus_idf(corpus_dir)

    # Build section-body index
    bodies: dict[str, str] = {}
    for md_file in sorted(corpus_dir.glob("*.md")):
        for section in parse_markdown(md_file, source_id=None):
            if section.content.strip():
                bodies[f"{md_file.name}#{slugify(section.heading)}"] = section.content

    paraphrases = load_paraphrases(queries_path)
    rekeyed: list[Paraphrase] = []
    missing: list[str] = []

    for p in paraphrases:
        body = bodies.get(p.gold_docs_section_id, "")
        if not body:
            missing.append(p.paraphrase_id)
            rekeyed.append(p)  # keep original tokens; warn below
        else:
            rekeyed.append(rekey_paraphrase(p, body, idf, top_n=top_n))

    if missing:
        # One-off script stdout is acceptable (CODING_STANDARD §5.1).
        print(
            f"WARNING: {len(missing)} Paraphrase(s) have a Gold Section id not "
            f"found in corpus — kept original Key Tokens: {missing}",
            file=sys.stderr,
        )

    # Re-use the existing YAML renderer; cost_usd is "n/a (offline re-key)" for
    # this non-LLM pass so the metadata block is honest (issue #104 cost-honesty).
    write_text_atomic(
        queries_path, render_queries_yaml(rekeyed, cost_usd="n/a (offline re-key)")
    )
    return rekeyed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI wrapper for ``rekey_queries``.

    Reads ``queries.yaml``, derives deterministic IDF Key Tokens for every
    Paraphrase, and atomically overwrites the file.  Prints a summary of
    multi-word elimination and empty-set check to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Re-key queries.yaml Key Tokens using deterministic corpus IDF.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Top-N IDF tokens per Gold Section body (default: %(default)s).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=QUERIES_PATH,
        help="Path to queries.yaml (default: committed set).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help="Path to the corpus directory (default: eval/paraphrase_comparison/corpus).",
    )
    args = parser.parse_args(argv)

    print(f"Re-keying {args.queries.name} from corpus at {args.corpus_dir} …")
    rekeyed = rekey_queries(args.queries, args.corpus_dir, top_n=args.top_n)

    # Summary stats
    multi_word = sum(1 for p in rekeyed for tok in p.key_tokens_docs if " " in tok)
    empty = sum(1 for p in rekeyed if not p.key_tokens_docs)
    print(
        f"Re-keyed {len(rekeyed)} Paraphrases. "
        f"Multi-word tokens: {multi_word} (was 250). "
        f"Empty Key-Token sets: {empty} (was 10)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
