---
status: accepted
---

# ADR-0039: Reader chat gains a demo-scoped guided-teaching layer

The reader `/chat` surface (`gateway/static/index.html`) now renders, under each
answer, a **curator note** that pairs the grounding outcome with the console
remediation that would address it — keyed on the server-provided `grounding.reason`
(plus the `filed` flag) and tagged with the console lint-check coordinate
(`C1` coverage, `C8` promotion, …) so the same code appears identically on the
reader surface and in the Operator Console. The empty state also carries a compact
"curation loop" catalog of the C1–C12 checks by [[lint-axis]].

This deliberately deviates from `CODING_STANDARD` §12.5's clean-reader posture (the
reader surface carries no explanatory scaffolding): a weak or withheld answer should
read as *"here's how a curator fixes this"* rather than a dead end, which is the
enterprise-KB-governance thesis the demo exists to convey. It reuses the existing
`[Source: …]` citation grammar — every curation action "cites" its `C#` coordinate.

## Considered options

- **Leave the reader clean, teach only in the console** — rejected: the evaluator's
  first touch is `/chat`, and a cold `Cannot Confirm` on a covered-looking topic
  reads as "broken" before they ever reach the console.
- **A loud info-callout / banner** — rejected as off-brand for the quiet editorial
  surface; the note is rendered as faint marginalia in the sans (human) voice, led
  by the mono coordinate, never louder than the answer.

## Consequences

- The teaching copy is **static, selected by the server code** (exactly like
  `REASON_TEXT`), so §12.4/§12.5's "no client business logic" still holds — nothing
  new is computed on the client beyond a lookup.
- **Demo-scoped.** A production reader deployment would gate it off behind a `demo`
  flag; the note is not part of the product's permanent reader contract.
- **Reusable, not fixture-bound.** The outcome→remediation mapping keys on live
  grounding/lint results, so it teaches against *any* corpus an operator loads — a
  visitor's own imported data triggers the same `C1`/`C8` notes on their own gaps
  and drafts, and stays silent when there is nothing to fix.

## Follow-up (reader zh/en toggle, follow-ups, C10 wire)

Live demo feedback surfaced three gaps this layer closes:

- **Reader language toggle.** The reader had no zh/en switch (only the Console
  did), and the Console's switch did not carry over. The reader now has its own
  masthead toggle persisted under the SAME `kb-console-lang` key the Console uses,
  so a switch on either surface follows to the other. This **supersedes** the
  earlier decision that the curator note follows the query's detected language: the
  note, verdict text, empty state, and all chrome now follow the explicit `uiLang`
  toggle. The answer BODY stays in the asked language (the model answers in the
  language of the question); only chrome is toggled. Chrome copy lives in a static
  `CHROME` map resolved via `t()` — still a lookup, no client business logic.
- **Follow-up chips.** After any answer the reader offers "ask next" suggestions so
  a first question opens onto a topic instead of a dead end. A coarse topic match
  on the interaction selects a curated, answerable set; unmatched topics fall back
  to the remaining presets (pinned answerable by the starter test). Static UI copy,
  bilingual by `uiLang`.
- **C10 on the wire.** The record-level `C10` tag (a schema-invalid Filed Answer
  surfacing as a source) is attached by retrieval but was dropped by the gateway's
  SSE source serializer, which forwards a fixed field set. The gateway now forwards
  `lint` when truthy (mirroring `derived_from`/`path`), so the reader can render the
  coordinate. Pinned by `test_chat_stream_wiki_forwards_c10_lint`.
