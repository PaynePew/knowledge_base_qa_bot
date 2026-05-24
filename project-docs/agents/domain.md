# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is **single-context**: one `CONTEXT.md` and one ADR folder at the root.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, if it exists.
- **`project-docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CLAUDE.md
├── CONTEXT.md                    ← created lazily by /grill-with-docs
├── project-docs/
│   ├── agents/                   ← this folder (issue-tracker, triage-labels, domain)
│   └── adr/                      ← architectural decisions (created lazily)
│       ├── 0001-...md
│       └── 0002-...md
├── docs/                         ← bot's runtime knowledge base, NOT project docs
├── scaffold/
└── PROMPT.md
```

> **Note** — root `docs/` holds the Q&A bot's runtime knowledge-base content (`account_help.md`, `refund_policy.md`, `shipping_faq.md`). It is the bot's input data, not project/architecture documentation. Keep ADRs and agent metadata out of `docs/`; they belong in `project-docs/`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
