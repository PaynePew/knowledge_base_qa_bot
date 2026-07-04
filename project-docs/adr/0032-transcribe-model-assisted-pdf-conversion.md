# Transcribe: model-assisted PDF conversion as a distinct operation, not an Import extension

A PDF without a text layer (the `NoTextLayer` rejection introduced by ADR-0031 / PRD #414) gets a supported path into the KB via a **new named operation, Transcribe**: rasterize pages (pypdfium2, already a transitive dependency), send each page image to a vision model under a strict faithful-transcription prompt, and write the assembled Markdown to `docs/` with the standard provenance envelope plus `origin: transcribed` and `transcribe_model`. Scoped in the 2026-07-04 grill of issue #419, after the knowledge owner confirmed the real corpus is scan-heavy and a real-artifact check showed even digital-native designed PDFs extract degraded.

The routing rule is **deterministic-only**: a cheap text-layer probe (pdfplumber) in front of the shared conversion entry routes text-less PDFs to Transcribe automatically; digital-native PDFs default to the free mechanical Import path, with a **manual per-file force** (`kb transcribe <path>`, `POST /transcribe`) as the designed-PDF escape hatch. Quality heuristics that auto-upgrade borderline digital files were rejected — they make Import's outcome unpredictable in both directions.

Why a new operation instead of extending Import: CONTEXT.md defines Import as "mechanical — No LLM calls; format conversion only", and that invariant is load-bearing (deterministic output, no network, no secrets, no per-page cost in the mechanical path). Transcribe is model-assisted and therefore none of those things. It is also **not Ingest**: the transcription prompt forbids summarization, synthesis, and completion — the model converts form (pixels → Markdown), it does not create content. The operation sits beside Import as a second `raw/ → docs/` converter, following the Capture precedent (ADR-0017) of a distinct operation writing `docs/` with its own provenance marker.

Key design points:

- **Provider**: GPT vision through the existing OpenAI key and the existing `ChatOpenAI` wrapper layer (ADR-0005 leaf-node borrowing). `OPENAI_TRANSCRIBE_MODEL` defaults to `gpt-5-mini` (~$2 per 1k pages). Zero new vendor, zero new secret, zero new SDK. Mistral OCR 3 (document-parsing specialist, strongest table reconstruction, $1–2/1k) and Gemini Flash (best VLM table scores, $5–8/1k) were rejected for v1 solely on integration cost — provider swap is a leaf-node change and both are recorded as upgrade paths if table fidelity on real scans disappoints.
- **Idempotency**: hash-skip keys on the raw PDF bytes, exactly like Import — so the nondeterminism of model output never causes rework; an unchanged file is never re-transcribed (and never re-billed).
- **Guards, fail-closed**: `KB_TRANSCRIBE_MAX_PAGES` (default 50) rejects oversized jobs with a typed failure before any model call; a missing API key is a typed unavailable failure; a page that still fails after bounded retry fails the whole file (`TranscribeError`) with no partial `docs/` write (Import's atomic-write convention).
- **Governance**: `origin: transcribed` + `transcribe_model` in the frontmatter keep model-derived Sources distinguishable from mechanical Imports and from Captures — the same reasoning as Capture's mandatory `origin: mcp-conversation`.
- **LLM surface accounting**: Transcribe adds one LLM call site, registered in ADR-0005 §"LLM-facing surface enumeration" with exactly one `@pytest.mark.live` smoke test (CODING_STANDARD §6.4).

## Considered Options

- **Extend Import with an OCR mode.** Rejected: breaks the "No LLM calls" invariant that makes Import deterministic and free; every downstream consumer of the Import contract would inherit a nondeterministic, billed, network-dependent mode.
- **Quality-probe auto-routing (auto-upgrade degraded digital extractions).** Rejected: heuristic thresholds (headings count, chars/page, radical-contamination ratio) misfire in both directions and make the ingest outcome unpredictable; the manual force covers the designed-PDF case with curator judgment.
- **Route everything through the model tier** (the "只推 Gemini/GPT" position). Rejected: right for magazine-grade corpora, wasteful here — the mechanical path is free, instant, and correct for genuinely digital-native files; ADR-0031's shipped machinery would be dead code.
- **Stage `raw/*.md` and reuse Import's passthrough** instead of writing `docs/` directly. Rejected: the `.pdf` origin falls off the provenance chain, and Import's hash-skip would key on model output bytes, which vary per run — idempotency breaks exactly where it matters most (billed re-runs).
- **Mistral OCR / Gemini as v1 provider.** Rejected for v1 on integration cost only (new vendor + secret + SDK vs. reusing the OpenAI stack); recorded as leaf-node upgrade paths.
- **Local OCR (docling+OCR, MinerU, Tesseract).** Still rejected for the VPS on ADR-0031 constraint 2 (torch-class weight); unchanged.

## Consequences

- New operation vocabulary: **Transcribe** (CONTEXT.md term; `kb transcribe`, `POST /transcribe`, MCP `kb_transcribe_v1` per ADR-0017 parity).
- The `NoTextLayer` failure stops being a dead end: the probe auto-routes scans to Transcribe at the shared entry, and the failure message (when Transcribe is unavailable) points at it.
- Per-page cost enters the system for scanned files only, bounded by the page cap and the hash-skip; at confirmed volumes this is dollars per month inside the ≤$15/mo posture.
- Transcribed Sources are auditable and filterable via `origin: transcribed`; lint/governance can treat them as a distinct class later without schema change.
- The Kangxi-radical codepoint normalization for MarkItDown output (found in the 2026-07-04 real-artifact check) is deliberately NOT part of Transcribe — it is a deterministic Import post-processing fix, filed separately.
