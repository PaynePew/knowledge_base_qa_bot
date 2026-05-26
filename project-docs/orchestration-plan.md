# Orchestration plan

How a fresh Claude Code session drives `ready-for-agent` issues to a merged PR. This document is the **meta-doc** — it points to the four per-role prompt files under [`project-docs/agents/`](agents/) and defines the loop, hand-off contract, and stop conditions. It deliberately does NOT contain agent prompts inline — those live in the role files so each can be tuned independently and stay context-light when injected into sub-agent prompts.

## Roles

| Role | Prompt file | Triggered by | Produces |
|---|---|---|---|
| **Plan** | [`agents/plan.md`](agents/plan.md) | Human or orchestrator at the start of a cycle | A `<plan>` JSON block: `top` (next issue to tackle) + `alternatives` + `blocked` |
| **Implement** | [`agents/implement.md`](agents/implement.md) | Orchestrator after Plan returns a `top` | A series of incremental commits on `slice/<N>-<desc>` + a structured COMPLETE / BLOCKED report on the issue |
| **Review** | [`agents/review.md`](agents/review.md) | Orchestrator after Implement returns COMPLETE | Zero or more `refactor:` commits + a PASS / PASS_WITH_CONCERNS / FAIL report on the issue. Injects relevant sections of `project-docs/CODING_STANDARD.md`. |
| **Merge** | [`agents/merge.md`](agents/merge.md) | Orchestrator after Review returns PASS or PASS_WITH_CONCERNS | Branch pushed to origin, PR opened with `Closes #N`, issue comment linking to the PR |

The **orchestrator** is the top-level Claude Code session (typically Opus 4.7). Sub-agents are spawned via the `Agent` tool — recommended models:

- Plan: Haiku 4.5 or Sonnet 4.6 (light reasoning, cheap)
- Implement: Sonnet 4.6 (main coding work)
- Review: Opus 4.7 (deeper reasoning, catches drift the implementer missed)
- Merge: Haiku 4.5 (mechanical workflow)

Model choice is a tuning knob — override per slice if a particular issue warrants it.

## The loop

```
[human]
  "run plan agent"
    ↓
[Plan agent — agents/plan.md]
  Reads: open ready-for-agent issues
  Outputs: <plan> JSON
    ↓
[orchestrator]
  Confirms top with human (Mode C — Manual gate)
    OR auto-advances (Mode B — Semi-auto, non-blocking)
    ↓
[Implement agent — agents/implement.md]
  Branch: slice/<N>-<desc> (orchestrator creates and checks out before spawn)
  TDD RGR per AC, incremental commits
  Posts COMPLETE or BLOCKED report on issue
    ↓
[Review agent — agents/review.md]
  Reads diff + commits + implementer report
  Injects relevant CODING_STANDARD.md sections (§3.1, §4, §5, §11 mandatory; others on demand)
  Posts PASS / PASS_WITH_CONCERNS / FAIL report on issue
    ↓ (if PASS or PASS_WITH_CONCERNS)
[Merge agent — agents/merge.md]
  Re-runs tests + ruff
  Pushes branch to origin
  Opens PR with Closes #N
  Comments on issue with PR link
    ↓
[human]
  Reviews PR on GitHub, merges
    ↓
GitHub auto-closes issue via Closes #N keyword
    ↓
[orchestrator]
  Returns to Plan agent for next slice OR stops if no more ready-for-agent issues
```

## Modes

- **Mode B — Semi-auto, non-blocking**: orchestrator advances Plan → Implement → Review → Merge → next without pause. Human can interrupt at any time (Esc Esc / Ctrl+C). Default for routine slices.
- **Mode C — Manual gate**: orchestrator stops after each agent and waits for the human to type "continue" (or equivalent). Default when human wants to inspect each step (e.g. first slice of a new project, or after a previous slice had concerns).

The human can request the mode switch at any time during the loop.

## Hand-off contract

### Issue labels

- `ready-for-agent` — Plan picks from this set. Set by `/to-issues` or human after triage.
- `needs-info`, `needs-triage`, `ready-for-human`, `wontfix` — Plan ignores these.

### Branch naming

`slice/<N>-<short-kebab-description>`. Examples: `slice/9-lock-design-docs`, `slice/10-grounding-foundations`.

The orchestrator creates and checks out the branch **before** spawning Implement. Implement does not create branches.

### Commit conventions

Conventional Commits per `~/.claude/rules/git-workflow.md`:

```
<type>: <description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `style`.

Slice commit bodies include: 2-3 sentence summary, AC checklist, files-touched list. See `agents/implement.md` for the exact template.

### PR body

Built by Merge agent. Always starts with `Closes #N`. Includes: implementer's What-was-built, AC self-report, reviewer's verdict + flagged concerns, final test result.

### Issue auto-close

The `Closes #N` keyword in the PR body makes GitHub close the issue automatically when the human merges. **No agent runs `gh issue close`.**

## Resume protocol

Branches may resume across sessions. Each agent checks the working tree on startup:

- If working tree is clean → start fresh per its prompt.
- If working tree is dirty → WIP-commit (`git commit -am "wip: checkpoint before resume"`) or stash, then proceed. Never `git reset --hard` silently. Per `agents/implement.md`.

The orchestrator can resume any in-flight slice by checking out the existing branch and re-spawning the agent at the right step.

## Stop conditions

Stop and surface to the human if any of the following occurs:

1. **Plan agent reports no ready-for-agent issues** — cycle complete.
2. **Implement agent returns BLOCKED twice in a row** on the same issue — deadlock; needs human intervention.
3. **Review agent returns FAIL twice in a row** — the implementer cannot satisfy the AC; needs human re-scope or specification fix.
4. **Merge agent returns BLOCKED** — pre-push tests fail in a way ruff cannot auto-fix; needs a fix-up implement pass.
5. **An agent proposes modifying `PROMPT.md`, `CONTEXT.md`, `project-docs/adr/`, `project-docs/inspiration.md`, or `project-docs/CODING_STANDARD.md`** that is NOT explicitly authorised by the slice AC — those are human territory; stop and surface for approval. (Slices that are docs-only and explicitly authorise such edits — e.g. Grounding Check Slice #1 — are the exception.)
6. **Any agent reports the venv is broken or a critical dependency is missing** — fix the environment before resuming.

## Spawning sub-agents (orchestrator-side guide)

When spawning a role agent, pass the prompt file path + a minimal substitution map. Recommended `Agent` tool invocation shape:

```
Agent(
  description="<slice> — <role>",
  subagent_type="general-purpose",
  model="sonnet",  # or opus / haiku per role
  prompt="""
You are the {ROLE} agent.

Read your role contract from: project-docs/agents/{ROLE}.md

Substitutions to apply when you see {{VARIABLE}} placeholders:
- {{ISSUE}} = <issue number>
- {{BRANCH}} = slice/<N>-<desc>
- {{TARGET_BRANCH}} = main
- {{IN_PROGRESS_LIST}} = <comma-separated, plan only>

Begin by following the prompt file's start-up sequence.
""")
```

Do NOT inline the full role prompt into the spawning message — that doubles the context cost. The sub-agent reads its own role file once on launch.

## Context economy

The orchestrator's job is to **move issues forward**, not to deeply re-understand the codebase each cycle. Per-cycle context for the orchestrator stays bounded:

- `gh issue list` results (Plan input)
- `<plan>` JSON output (Plan → Implement hand-off)
- Issue body excerpts when relevant
- Final reports from each agent

Sub-agents rebuild their own context from the repo (their role file + the docs they need). Do NOT paste the orchestrator's conversation history into a sub-agent prompt.

## After the last ready-for-agent issue is merged

Plan agent returns `{"top": null, "alternatives": [], "blocked": []}` (or just blocked items). The orchestrator reports:

> All ready-for-agent issues processed. Blocked: [list]. Recommend running `/grill-with-docs` on Phase 2 design before generating the next batch of issues.

## What the orchestrator does NOT do

- Change `PROMPT.md`, `CONTEXT.md`, ADRs, `CODING_STANDARD.md`, or `inspiration.md` (those are human-authored or `/grill-with-docs`-driven).
- Create new issues from scratch — `/to-prd` + `/to-issues` are the human-invoked paths.
- Run `git push` to `main` — Merge agent pushes the slice branch; humans merge PRs on GitHub.
- Run `pytest -m live` automatically — that requires the human's `OPENAI_API_KEY` and is a manual verification step before declaring a slice production-ready.
- Touch `.claude/` or any harness configuration.
