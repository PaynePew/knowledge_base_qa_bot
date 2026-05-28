# Paraphrase Comparison Report

Phase 8 retrieval comparison (PRD #100). Stack A = Wiki + BM25; Stack B = Vector RAG. Numbers are the deterministic C5c hit metric (source-match AND Key-Token overlap).

Stack B embedding mode: **fake** (`fake` = deterministic offline stand-in used when OPENAI_API_KEY is absent; `real` = OpenAI `text-embedding-3-small`).

| Paraphrase Type | Stack A hit_rate@3 | Stack B hit_rate@3 |
|---|---|---|
| implicit_reference | 0.875 | 0.750 |
| industry_jargon | 0.800 | 1.000 |
| specificity_narrowing | 1.000 | 1.000 |
| synonym_swap | 0.375 | 0.250 |
| typo_fatfinger | 0.200 | 0.000 |
| verbosity_expansion | 1.000 | 0.750 |
| word_reorder | 0.875 | 1.000 |
