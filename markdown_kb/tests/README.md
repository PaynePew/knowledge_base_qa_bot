# Test Philosophy — Markdown KB

This app uses an integration-first pytest setup. The test pyramid is intentionally inverted compared to a strict-TDD codebase: thick at the integration layer, thin at unit. The PROMPT.md verification cases are the contract; integration tests at the FastAPI `TestClient` level translate that contract one-to-one into executable assertions, so "all green" means "the deliverable is met."

## Layout

```
tests/
├── README.md                   ← this file
├── conftest.py                 ← shared fixtures: client, fake LLM, tmp .kb
├── fixtures/
│   └── docs/                   ← mini Markdown corpus, deliberately matches docs/
├── test_indexing.py            ← component: parse_markdown, build_index, BM25
├── test_chat_grounded.py       ← integration: PROMPT.md curl cases
├── test_chat_fallback.py       ← integration: Cannot Confirm paths
└── test_chat_live.py           ← @pytest.mark.live — opt-in real OpenAI smoke
```

## Why integration-first

- The PROMPT.md curl cases *are* the spec — translating them into pytest gives a single source of truth for "done."
- The real failure surface is the full chain (parse → index → BM25 rank → prompt build → LLM call → response shape). Unit tests on `slugify` would not catch the bugs that actually happen.
- The LLM is mocked by default (the fake LLM returns a canned response shaped like a real grounded answer). This makes tests deterministic, free, and CI-safe. The mocked LLM still receives the real prompt, so prompt-builder regressions are caught.

## Why a live smoke test exists

Mocking the LLM cannot tell us whether the real model actually follows our system prompt (cites correctly, refuses out-of-scope queries, etc.). One `@pytest.mark.live` test makes a real OpenAI call and asserts only the response shape — not specific words — so it stays robust across model updates. Run with `pytest -m live` before pushing; skipped by default in CI.

## What is *not* tested

- `slugify()` and other trivial helpers — covered transitively by the component tests on `parse_markdown`.
- BM25 score absolute values — only ranking order is asserted, because BM25 numerics are sensitive to corpus parameters and would create brittle tests.
- LLM output content beyond shape and presence of expected `[Source: ...]` citations.
