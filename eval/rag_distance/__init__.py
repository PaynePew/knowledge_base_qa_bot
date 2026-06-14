"""RAG (Stack B) pre-LLM distance-gate calibration (#257 / #258 follow-up).

Sibling of ``eval.negative_case`` (which calibrates the wiki BM25 ``KB_SCORE_THRESHOLD``).
This package calibrates the RAG FAISS distance ceiling ``KB_RAG_DISTANCE_THRESHOLD``
shipped OFF in #258: it drives the REAL ``vector_rag`` index + ``search_with_distance``
over the same committed in-scope corpus + positive/negative query sets as the
negative-case eval, so the two stacks' gates are calibrated on identical data and
directly comparable.

Unlike the BM25 sweep it requires REAL embeddings (``text-embedding-3-small``), so
``calibrate.main`` is a manual, quota-spending run — the hermetic tests use the
deterministic offline fake embeddings instead.
"""
