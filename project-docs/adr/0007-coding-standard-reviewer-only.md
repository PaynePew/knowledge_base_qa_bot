# CODING_STANDARD as reviewer-only abstract rules

`project-docs/CODING_STANDARD.md` is restricted to reviewer-time injection only — implementer agents and human contributors writing code never read it. The file's content is constrained to F-strict form: rule prose, internal §X.Y self-references, references to other meta-docs (CONTEXT.md, ADR-N, log-kinds.md, PRD), and CONTEXT.md vocabulary terms used as concepts. Module names, function names, constant names, class names (except where they exactly match CONTEXT.md vocabulary), code-tree paths inside `markdown_kb/`, and phase-specific inventories are banned. Anchors that previously lived here migrate to two homes: ADRs carry invariant-tagged code anchors via §2.5 grep flow, and `agents/implement.md` carries an implementer-side cheat sheet that mirrors the highest-frequency drift signals in F-strict form. The cheat sheet is curated through grill sessions, not auto-synced.

We chose this design because the previous CODING_STANDARD mixed two grains of content with incompatible lifetimes — timeless rules (vocabulary discipline, deep modules, fail-fast on corruption, single log channel) and module-specific anchors (`see parse_markdown in indexer.py`, `current LLM-facing set is retrieval.py and grounding.py`, `After Phase 3 the surfaces are /chat and /ingest`). The latter rot every slice; the former should never need editing. Mixing them produced repeated end-of-phase rot — the Phase 3 → 4 transition produced 5 stale claims across §4.3, §6.1, §6.4, §6.5, §2.5 that no slice's reviewer caught, even with §11 walks running on every PR. Separating by grain — and by reader — eliminates the rot vector while preserving the rules' enforcement authority.

## Considered Options

### Where do module-specific anchors go after extraction (Q1)

- **A — Sibling file `project-docs/CODING_EXAMPLES.md`**: a new mutable file declares itself "expected to drift". Rejected: relocates the drift problem without solving it; reviewer now injects two files and breaks the §0.2 ~80-line budget; new file becomes the next forgotten doc.
- **B — Inline in each ADR**: every invariant carries its own code-site example in ADR § Consequences. Partial fit — works for ADR-derived invariants but doesn't cover non-ADR rules (sentinel string discipline, §10 pattern recognition).
- **C (chosen) — hybrid pure-rules CODING_STANDARD + ADR-side invariant anchors + implementer cheat sheet, no in-code annotations**: rules-only CODING_STANDARD reviewer-only; ADRs anchor invariants in § Consequences; `agents/implement.md` carries a short anti-pattern cheat sheet in F-strict form. No `# CS §X.Y` comments — code stays clean.
- **D — Drop all anchors entirely**: rules-only with no examples anywhere; reviewer grep-discovers everything. Rejected: removes pedagogical anchor that newcomers (human and AI) need to map abstract rules to concrete sites.

### Who reads CODING_STANDARD (Q2)

- **(i) Implementer + reviewer (status quo)**: §0 asks implementer to load it on first session. Rejected: empirical evidence from Phase 4 shows pre-emptive reading does not produce compliance — Slice 4-5a #50 introduced the §11 `indexer.search` mock despite the file being in the implementer's reading list. Reviewer FAIL is the actual enforcement gate.
- **(ii) Reviewer only (chosen)**: §0.2 already declared this as *intent*; this ADR makes it the *only* contract. Implementer reads vocabulary (CONTEXT.md), ADRs, the issue body, and the `agents/implement.md` cheat sheet. Trade-off: implementer may generate code that violates abstract rules more often, costing additional fixup cycles. Phase 4 measurement: 2 fixup cycles across 7 slices (~28% rate) is the cost baseline; the simplification's value exceeds it.

### Code-side annotation (Q3)

- **(a) `# CS §X.Y` comments on canonical sites**: reviewer greps `# CS §3.3` to find canonical site. Rejected: violates §1.8 (`# CS §3.3` is metadata, not WHY); annotations themselves drift when code moves; LLM reviewer benefits little (grep on constant name is already fast).
- **(b) No code annotations (chosen)**: reviewer §11 walk uses naked grep against code patterns. Empirically all Phase 4 §11 catches were via grep, not annotation.

### Anti-drift enforcement (Q4)

- **(α) Pure human discipline**: relies on every human edit to remember the F-strict line. Rejected: produced the 5 stale claims this phase.
- **(β) CI guard — single grep test (chosen)**: one pytest assertion that CODING_STANDARD.md contains no `.py` reference. Catches the highest-frequency drift category mechanically; other categories remain on reviewer judgment. The test costs ~5 lines and ~10ms per CI run.
- **(γ) Full lint suite**: regex against all banned categories (constant names, code paths, phase inventories). Rejected: false-positive prone; maintenance burden exceeds benefit at this scale.

### Migration (Q5)

- **One slice landing ADR + F-strict rewrite + cheat sheet update + CI guard (chosen)**: mirrors Slice 4-1's docs-slice pattern. Explicit AC overrides stop-condition #5 for the touched human-territory files.
- Multi-slice (ADR first, rewrite second): adds latency without design benefit; the ADR fully constrains the rewrite.

## Consequences

**For CODING_STANDARD.md content:**
- §0 reading order item 5 narrows to "when reviewing code".
- §1.5 / §1.6 / §2.4 / §2.6 / §3.3 / §4.2 / §10 / §11 — module names / function names / constant names / code paths abstracted out.
- §4.3 stale "one retry at temperature=0 / clear sources" line deleted; impl detail defers to ADR-0001 / ADR-0004.
- §6.1 ↔ §6.4 contradiction resolved — "exactly one live test" replaced with "one per LLM-facing surface; current surfaces enumerated in the relevant ADR § Consequences".
- §10 Patterns table loses the "Where" column.
- §11 drift signals — `indexer.search` → "deep-module entry points"; `retrieval.py` LangChain anchor → "LLM-call wrapper module"; etc.
- §6.5 fixture path abstracted; principle (deterministic hand-written fixtures) stays.
- §8 Tooling and §9 Commits substantially unchanged.

**For agents/implement.md:**
- Drop "Coding standard" lazy-read item; drop §8.2 mypy back-reference.
- Rewrite 8-bullet "Project-specific anti-patterns" in F-strict form; stale `SOURCE_DIRS` bullet deleted; "one smoke test" bullet updated to "one per LLM-facing surface"; remove all `Per CODING_STANDARD.md §X.Y` back-references.
- Stop-condition #5 protection unchanged — agents cannot edit CODING_STANDARD without explicit AC authorisation.

**For agents/review.md:**
- §0.2 lazy-injection contract unchanged in shape (mandatory §3 + §4 + §5 + §11; conditional §1 / §2 / §6 / §7 / §10). Section content shifts to F-strict; injection mechanism the same.
- §11 walk continues to use grep against code patterns; no annotation lookup added.

**For future grill sessions:**
- A new type β rule discovered during grill produces: (a) a new ADR with invariant + code anchor in § Consequences, (b) possibly a CODING_STANDARD rule addition in F-strict form, (c) possibly an implement.md cheat sheet addition. Grill curates all three; no automatic sync.

**For the orchestration loop:**
- The follow-up slice's AC must explicitly authorise the four human-territory files touched: ADR-0007, CODING_STANDARD.md, agents/implement.md, agents/review.md. Precedent: Slice 4-1 / ADR-0006.
