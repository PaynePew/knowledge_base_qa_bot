# Paraphrase Comparison Report — Pending Real Run

The real three-arm eval numbers (Stack A / B / C with genuine `text-embedding-3-small`
embeddings and gpt-4o-generated paraphrases) are **pending the #317 real-embedding run**.
This file is a pointer stub; it contains no measurement data.

## What lives where

| Artifact | Description |
|---|---|
| `report.offline-tracer.md` | Pipeline-validation tracer — `--fake-embeddings` run. Dense arms use deterministic stand-ins (token-overlap / SHA-256 hash vectors). **Do not interpret these numbers as real measurements.** |
| `charts-offline/` | Charts for the tracer run above. |
| `report.md` *(this file)* | Stub pending the real three-arm eval (#317). |
| `charts/` | Will be written by the real-embedding run. |

## Recovering the prior two-arm real report

The real Phase 8 two-arm numbers (gpt-4o paraphrases + `text-embedding-3-small`,
Stack A 0.912 vs Stack B 0.940) are recoverable from git:

```
git show c7747a3:eval/paraphrase_comparison/report.md
```

Those numbers used the Phase 8 single-cutoff methodology; the upcoming real
three-arm run (#317) supersedes them with the Phase 13 cutoff-sweep + Cochran's Q
omnibus methodology.
