"""One-off CLI to run the Phase 8 retrieval comparison and write report.md.

Usage (from repo root):

    uv run python -m eval.paraphrase_comparison.run_comparison

With OPENAI_API_KEY set, Stack B uses real ``text-embedding-3-small`` vectors.
Without a key, pass ``--fake-embeddings`` to run a deterministic offline
stand-in (token-overlap ranker) so the loop is still exercisable; the report
records which mode produced the numbers.

The opt-in L2 cross-family **Spot-check** is enabled with ``--judge=<model>``
(default ``claude-sonnet-4-6``; documented choices ``claude-haiku-4-5`` /
``claude-sonnet-4-6`` / ``claude-opus-4-7``). It requires ``ANTHROPIC_API_KEY``
— without ``--judge`` the Spot-check is skipped and the report notes how to
enable it; with ``--judge`` but no key the run fail-fasts with a clear message.
Zone tuning: ``--judge-zones``, ``--judge-marginal-threshold`` (default 1),
``--judge-control-sample-size`` (default 5).

This is a one-off script, so a stdout summary via ``print`` is acceptable
(CODING_STANDARD §5.1 — the no-print rule is scoped to committed library code).
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv

from . import runner as _runner
from .runner import JudgeConfig, run_comparison
from .spotcheck import (
    DEFAULT_CONTROL_SAMPLE_SIZE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MARGINAL_THRESHOLD,
    JUDGE_MODELS,
    ZONES,
    JudgeUnavailableError,
)


def _install_fake_embeddings() -> None:
    """Install deterministic offline stand-ins for the dense arms AND the reranker.

    Stack B (vector_rag): swap its FAISS factory for a token-overlap ranker.
    Stack C (hybrid_kb dense-over-wiki): swap its ``get_embeddings`` leaf for a
    deterministic hash-based ``Embeddings`` so the REAL FAISS build/search path
    runs offline (mirrors the eval test conftest's ``fake_dense_embeddings`` and
    the hybrid_kb suite). Stack C+rerank (ADR-0019): swap ``hk_rerank.get_cross_encoder``
    for a deterministic token-overlap cross-encoder so the offline tracer exercises
    the 4th arm without downloading the ~2.3 GB ``bge-reranker-v2-m3`` model. None
    touch the network, so the whole four-arm comparison is reproducible without
    ``OPENAI_API_KEY`` (and without the optional ``rerank`` dependency).
    """
    import hashlib
    from dataclasses import dataclass

    import hybrid_kb.app.dense_index as hk_dense
    import hybrid_kb.app.rerank as hk_rerank
    import vector_rag.app.indexer as vr_indexer
    from langchain_core.embeddings import Embeddings
    from markdown_kb.app.indexer import tokenize

    @dataclass
    class _FakeDoc:
        page_content: str
        metadata: dict

    class _FakeVectorStore:
        def __init__(self, documents):
            self._docs = [_FakeDoc(d.page_content, dict(d.metadata)) for d in documents]

        def similarity_search_with_score(
            self, query: str, k: int = 3, filter=None, fetch_k: int = 20
        ):
            # Mirror FAISS's dict-metadata filter (#290 RAG language filter): drop
            # docs whose metadata does not match every key in ``filter`` BEFORE
            # ranking, so the offline comparison reproduces the real same-language
            # retrieval rather than cross-language leaking. ``fetch_k`` is accepted
            # for signature parity but the in-memory fake ranks all docs anyway.
            docs = self._docs
            if filter:
                docs = [
                    d
                    for d in docs
                    if all(d.metadata.get(fk) == fv for fk, fv in filter.items())
                ]
            q = set(tokenize(query))
            scored = [(d, len(q & set(tokenize(d.page_content)))) for d in docs]
            scored.sort(key=lambda t: -t[1])
            return [(d, 1.0 / (1.0 + o)) for d, o in scored[:k]]

        def save_local(self, folder_path: str, index_name: str = "index") -> None:
            # vector_rag.build_index persists on success (issue #103); the fake is
            # in-memory only and the comparison never reloads, so persistence is a
            # harmless no-op. _isolate_production_paths still repoints FAISS_INDEX_DIR
            # to tmp, so even a real save would never touch production .kb/.
            return None

    vr_indexer._build_faiss = lambda documents: _FakeVectorStore(documents)

    class _FakeDenseEmbeddings(Embeddings):
        """Deterministic SHA-256-derived embeddings for Stack C's dense-over-wiki arm."""

        _DIM = 16

        def _vec(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            return [b / 255.0 for b in digest[: self._DIM]]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    _fake_dense = _FakeDenseEmbeddings()
    hk_dense.get_embeddings = lambda: _fake_dense

    class _FakeCrossEncoder:
        """Deterministic token-overlap stand-in for the rerank cross-encoder.

        Scores each (query, passage) pair by shared-token count via the same
        language-aware ``tokenize`` the offline dense fakes use (Latin words + CJK
        bigrams), so the offline 4th arm reorders bilingually without any model.
        """

        def predict(self, pairs):
            scores = []
            for query, passage in pairs:
                overlap = len(set(tokenize(query)) & set(tokenize(passage)))
                scores.append(float(overlap))
            return scores

    _fake_cross_encoder = _FakeCrossEncoder()
    hk_rerank.get_cross_encoder = lambda: _fake_cross_encoder


def _isolate_production_paths() -> None:
    """Redirect markdown_kb + vector_rag persisted-index/log paths to a temp dir.

    The pytest suite does this via an autouse fixture; the standalone CLI must do
    it too. Otherwise indexing Stack A/B writes the production ``.kb/index.json``
    and ``wiki/log.md`` at the repo root — which then pollutes hermetic tests that
    lazy-load the persisted index (#148/#133). Mirrors the test conftest's
    ``_redirect_markdown_kb_paths`` (issue #145).
    """
    import tempfile
    from pathlib import Path

    import hybrid_kb.app.dense_index as hk_dense
    import hybrid_kb.app.logger as hk_logger
    import markdown_kb.app.indexer as mk_indexer
    import markdown_kb.app.logger as mk_logger
    import vector_rag.app.indexer as vr_indexer
    import vector_rag.app.logger as vr_logger

    iso = Path(tempfile.mkdtemp(prefix="phase8_cmp_iso_"))
    mk_indexer.INDEX_PATH = iso / ".kb" / "index.json"
    mk_indexer.WIKI_DIR = iso / "wiki"
    mk_logger.LOG_PATH = iso / "wiki" / "log.md"
    vr_indexer.FAISS_INDEX_DIR = iso / ".kb" / "faiss_index"
    vr_logger.LOG_PATH = iso / "vector_rag" / "log.md"
    # Stack C's dense-over-wiki seed (.kb/hybrid_dense/) must be isolated too, so
    # the standalone CLI never overwrites the committed seed (#316 / #307 lesson).
    hk_dense.DENSE_INDEX_DIR = iso / ".kb" / "hybrid_dense"
    hk_logger.LOG_PATH = iso / "hybrid_kb" / "log.md"


def _judge_config(args: argparse.Namespace) -> JudgeConfig | None:
    """Build the opt-in L2 Spot-check config from the ``--judge*`` flags (or None).

    ``--judge`` is the opt-in switch: absent -> None (Spot-check skipped). When
    present, ``--judge`` may be bare (use the default model) or carry a model
    name. Zone selection is parsed from ``--judge-zones`` (comma-separated).
    """
    if args.judge is None:
        return None
    model = args.judge or DEFAULT_JUDGE_MODEL
    zones = (
        tuple(z.strip() for z in args.judge_zones.split(",") if z.strip())
        if args.judge_zones
        else ZONES
    )
    return JudgeConfig(
        judge_model=model,
        zones=zones,
        marginal_threshold=args.judge_marginal_threshold,
        control_sample_size=args.judge_control_sample_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 retrieval comparison runner.")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use a deterministic offline embedding stand-in (no API key needed).",
    )
    parser.add_argument("--k", type=int, default=3, help="hit_rate@k cutoff.")
    parser.add_argument(
        "--judge",
        nargs="?",
        const=DEFAULT_JUDGE_MODEL,
        default=None,
        choices=JUDGE_MODELS,
        metavar="MODEL",
        help=(
            "Enable the opt-in L2 cross-family Claude judge Spot-check. Bare flag "
            f"uses {DEFAULT_JUDGE_MODEL}; choices: {', '.join(JUDGE_MODELS)}. "
            "Requires ANTHROPIC_API_KEY (fail-fast if absent)."
        ),
    )
    parser.add_argument(
        "--judge-zones",
        default=None,
        metavar="Z1,Z2,...",
        help=f"Comma-separated Spot-check zones (default: {','.join(ZONES)}).",
    )
    parser.add_argument(
        "--judge-marginal-threshold",
        type=int,
        default=DEFAULT_MARGINAL_THRESHOLD,
        help="Max Key-Token overlap for the Marginal zone (default 1).",
    )
    parser.add_argument(
        "--judge-control-sample-size",
        type=int,
        default=DEFAULT_CONTROL_SAMPLE_SIZE,
        help="Clear-hit/clear-miss count for the Control zone (default 5).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Skip the 4th arm (Stack C + cross-encoder reranker, ADR-0019). The "
            "reranker runs by default; a REAL run needs the optional dependency "
            "(`uv sync --group rerank`), an offline run uses a deterministic fake."
        ),
    )
    args = parser.parse_args(argv)
    load_dotenv(
        find_dotenv(usecwd=True)
    )  # pick up OPENAI_API_KEY from a repo-root .env

    fake = args.fake_embeddings or not os.getenv("OPENAI_API_KEY")
    mode = "fake" if fake else "real"
    if fake:
        _install_fake_embeddings()
        # Route offline runs to the trust-marked tracer artifact so they cannot
        # overwrite the canonical report.md / charts/ (CODING_STANDARD §6.6, #328).
        report_path = _runner.OFFLINE_TRACER_REPORT_PATH
        charts_dir = _runner.OFFLINE_CHARTS_DIR
        header: str | None = _runner.OFFLINE_TRACER_HEADER
    else:
        report_path = _runner.REPORT_PATH
        charts_dir = _runner.REPORT_PATH.parent / "charts"
        header = None

    # Never write the production .kb/ or wiki/log.md from the standalone CLI.
    _isolate_production_paths()

    judge = _judge_config(args)
    try:
        stack_a, stack_b = run_comparison(
            k=args.k,
            embedding_mode=mode,
            judge=judge,
            report_path=report_path,
            charts_dir=charts_dir,
            header=header,
            with_rerank=not args.no_rerank,
        )
    except JudgeUnavailableError as exc:
        # Opt-in Spot-check fail-fast: a clear one-line message + non-zero exit,
        # not a stack trace (issue #105 — flag set, key absent).
        print(f"error: {exc}")
        return 2

    print(f"Phase 8 comparison complete (Stack B embedding mode: {mode}).")
    if judge is not None:
        print(f"L2 Spot-check ran with judge {judge.judge_model}.")
    for ptype in sorted(set(stack_a.by_type) | set(stack_b.by_type)):
        a = stack_a.by_type.get(ptype, 0.0)
        b = stack_b.by_type.get(ptype, 0.0)
        print(f"  {ptype}: Stack A hit_rate@{args.k}={a:.3f}  Stack B={b:.3f}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
