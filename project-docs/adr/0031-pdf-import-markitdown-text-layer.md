# PDF Import: text-layer extraction via MarkItDown; scanned PDFs rejected

Import (the Phase 7 mechanical `raw/` → `docs/` conversion) gains `.pdf` through MarkItDown's text-layer extraction (a pdfplumber + pdfminer.six hybrid). Digital-native PDFs only — no OCR, no VLM, no network call: a PDF without a text layer fails with a typed `NoTextLayer` error carrying curator guidance, rather than converting into an empty Source. This closes Fork 3 of `project-docs/large-file-ingest-research.md` (deferred 2026-06-13 pending "its own focused pass"; the pass is the 2026-07-03/04 grill behind PRD #414).

Five constraints shaped the choice, in priority order:

1. **The Section model needs headings** — the retrieval unit is the heading-anchored Section, so heading emission outranks even table fidelity. An extractor that outputs flat text collapses an entire manual into one Section.
2. **Deploy image weight** — Import runs server-side, so the extractor ships inside the `python:3.11-slim` VPS image (≤$15/mo posture). Torch-class dependencies (2+ GB) are disqualifying for a core dep.
3. **Import's mechanical invariant** — CONTEXT.md defines Import as "No LLM calls; format conversion only". Cloud OCR/VLM extraction would smuggle network, per-page cost, API secrets, and nondeterminism into a deterministic path.
4. **CJK must extract losslessly** — the corpus is Chinese-heavy (ADR-0014 / Phase 16); acceptance is fixture-gated.
5. **License hygiene** — public repo serving over the network; copyleft surprises are not worth the marginal quality.

MarkItDown (MIT, ~80 MB, no torch, Microsoft, active — v0.1.6 May 2026) is the only candidate that clears all five. Its historically weak PDF tables were reworked in v0.1.5+ (aligned Markdown tables via the pdfplumber hybrid, wide-table support); the "MarkItDown 表格慘" reputation predates that and was earned on magazine-grade layouts this KB explicitly does not target.

This ADR also **amends ADR-0005's per-format mapping** for PDF. The original pre-blessing ("LlamaHub `PDFReader` or `pypdf`") predates the 2025–2026 extractor landscape: both emit flat, heading-less text, which defeats the Section retrieval model (constraint 1) — the very opinion ADR-0005 exists to protect.

## Considered Options

- **pymupdf4llm / PyMuPDF** — best lightweight-class quality (font-size heading inference, fast, good CJK), but **AGPL-3.0** (MuPDF, Artifex). Rejected on constraint 5 despite the technical fit being otherwise the strongest.
- **docling (IBM)** — best local table structure (TableFormer, TEDS >91% on FinTabNet), MIT, CPU-capable (~0.8 s/page median), built-in OCR. Rejected as the core dependency on constraint 2 (~2.4 GB torch + models). **Deferred as an opt-in quality tier** with explicit revisit triggers: (a) real table-fidelity failures on actual corpus PDFs, or (b) `docling-slim` maturing into a lightweight install.
- **marker (datalab)** — strong on academic/book layouts, but GPL-3.0 code plus revenue-capped RAIL-M model weights, plus torch. Rejected on constraints 2 and 5; no quality gain on this corpus.
- **MinerU (OpenDataLab)** — top open-source OmniDocBench scores, notably strong CJK; license is now Apache-2.0-with-terms (no longer AGPL). Rejected on constraint 2 (multi-GB torch/paddle stack). Noted as the fallback direction should MarkItDown's CJK extraction ever fail the fixture gate.
- **pdfplumber DIY** — MIT and light, but has no Markdown assembly; we would hand-roll heading heuristics, i.e. re-implement MarkItDown's PDF path at the joint instead of borrowing at the leaf (violates ADR-0005's borrowing principle).
- **pypdf / LlamaHub PDFReader** — ADR-0005's original pre-blessing. Flat text, no headings → single-Section collapse. Rejected on constraint 1; mapping line amended.
- **Cloud OCR / VLM services** — Mistral OCR ($2/1k pages; the free tier no longer exists), Gemini Flash (~$5–8/1k), GPT-5-mini (~$2/1k), Claude PDF input (~$7–10/1k on Haiku). Best-in-class on scans and free-form layouts, and the right answer for magazine-grade corpora — but this corpus is digital-native, and constraint 3 rules them out of Import categorically. **If scanned sources become a recurring real need, that is a new named operation with its own grill, not an Import extension**; the natural shape (cheap text-layer probe → route text-less docs to an OCR service) is recorded here for that future grill.
- **Self-hosted open VLMs (Qwen3-VL, MinerU2.5-Pro, PaddleOCR-VL)** — top benchmark scores but require GPU serving; non-starter on the VPS (constraint 2).

## Consequences

- `markitdown` (with its PDF extra) becomes a `markdown_kb` dependency and ships in the deploy image (~80 MB, MIT, no torch).
- Scanned/image-only PDFs fail `NoTextLayer`; encrypted PDFs fail `EncryptedPdf`; extractor crashes fail `PdfExtractionError` — three typed failure modes joining Phase 7's existing twelve (PRD #414, slices #415–#417).
- Heading fidelity is bounded by MarkItDown's font-based inference. A no-heading PDF degrades to a single Section (the `.txt` precedent), never rejected. Table fidelity is accepted at MarkItDown's current level; complex-table failures are the docling revisit trigger, not a bug.
- The extractor sits behind the importer's per-format converter seam, so a future quality-tier swap is a leaf-node replacement, not a redesign.
- CJK acceptance is fixture-gated in slice #415 and additionally verified pre-merge against a genuine externally-produced Chinese PDF over live HTTP (output not committed).
- ADR-0005's per-format mapping line for PDF now points here.
