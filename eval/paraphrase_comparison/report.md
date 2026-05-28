# Paraphrase Comparison Report

Phase 8 retrieval comparison (PRD #100). Stack A = Wiki + BM25; Stack B = Vector RAG. Numbers are the deterministic C5c hit metric (source-match AND Key-Token overlap).

Stack B embedding mode: **fake** (`fake` = deterministic offline stand-in used when OPENAI_API_KEY is absent; `real` = OpenAI `text-embedding-3-small`).

| Paraphrase Type | Stack A hit_rate@3 | Stack B hit_rate@3 |
|---|---|---|
| synonym_swap | 1.000 | 0.800 |
