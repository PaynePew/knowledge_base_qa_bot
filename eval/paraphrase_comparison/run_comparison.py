"""One-off CLI to run the Phase 8 retrieval comparison and write report.md.

Usage (from repo root):

    uv run python -m eval.paraphrase_comparison.run_comparison

With OPENAI_API_KEY set, Stack B uses real ``text-embedding-3-small`` vectors.
Without a key, pass ``--fake-embeddings`` to run a deterministic offline
stand-in (token-overlap ranker) so the loop is still exercisable; the report
records which mode produced the numbers.

This is a one-off script, so a stdout summary via ``print`` is acceptable
(CODING_STANDARD §5.1 — the no-print rule is scoped to committed library code).
"""

from __future__ import annotations

import argparse
import os
import sys

from .runner import run_comparison


def _install_fake_embeddings() -> None:
    """Swap vector_rag's FAISS factory for a deterministic token-overlap ranker."""
    from dataclasses import dataclass

    import vector_rag.app.indexer as vr_indexer
    from markdown_kb.app.indexer import tokenize

    @dataclass
    class _FakeDoc:
        page_content: str
        metadata: dict

    class _FakeVectorStore:
        def __init__(self, documents):
            self._docs = [_FakeDoc(d.page_content, dict(d.metadata)) for d in documents]

        def similarity_search_with_score(self, query: str, k: int = 3):
            q = set(tokenize(query))
            scored = [(d, len(q & set(tokenize(d.page_content)))) for d in self._docs]
            scored.sort(key=lambda t: -t[1])
            return [(d, 1.0 / (1.0 + o)) for d, o in scored[:k]]

        def save_local(self, folder_path: str, index_name: str = "index") -> None:
            # vector_rag.build_index persists on success (issue #103); the fake is
            # in-memory only and the comparison never reloads, so persistence is a
            # harmless no-op. _isolate_production_paths still repoints FAISS_INDEX_DIR
            # to tmp, so even a real save would never touch production .kb/.
            return None

    vr_indexer._build_faiss = lambda documents: _FakeVectorStore(documents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 retrieval comparison runner.")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use a deterministic offline embedding stand-in (no API key needed).",
    )
    parser.add_argument("--k", type=int, default=3, help="hit_rate@k cutoff.")
    args = parser.parse_args(argv)

    fake = args.fake_embeddings or not os.getenv("OPENAI_API_KEY")
    mode = "fake" if fake else "real"
    if fake:
        _install_fake_embeddings()

    stack_a, stack_b = run_comparison(k=args.k, embedding_mode=mode)

    print(f"Phase 8 comparison complete (Stack B embedding mode: {mode}).")
    for ptype in sorted(set(stack_a.by_type) | set(stack_b.by_type)):
        a = stack_a.by_type.get(ptype, 0.0)
        b = stack_b.by_type.get(ptype, 0.0)
        print(f"  {ptype}: Stack A hit_rate@{args.k}={a:.3f}  Stack B={b:.3f}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
