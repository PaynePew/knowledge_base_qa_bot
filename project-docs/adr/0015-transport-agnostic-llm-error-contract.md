# Transport-agnostic LLM-error contract

The LLM-call error-handling wrapper must surface failures in a way every interface can render. CODING_STANDARD §4.2 currently mandates that the wrapper raise `fastapi.HTTPException` directly (with a 503/500 status table). That mandate predates Phase 12, when HTTP was the only face. Phase 12 adds two non-HTTP interfaces (MCP, CLI), and a deep module raising an HTTP-framework exception leaks the transport into callers that have no HTTP status to render. This ADR makes the wrapper transport-agnostic.

## Decision

### The wrapper raises a domain error, not `HTTPException`

`_call_llm_with_error_handling` (in both `markdown_kb/app/retrieval.py` and `vector_rag/app/retrieval.py`) raises a new domain exception:

```python
# markdown_kb/app/errors.py
class LLMError(Exception):
    """Transport-agnostic LLM-call failure. Interface adapters render it."""
    def __init__(self, *, retryable: bool, message: str):
        self.retryable = retryable
        self.message = message
        super().__init__(message)
```

`LLMError` is defined once in `markdown_kb` and imported by `vector_rag` — the same cross-package seam `vector_rag` already uses for `grounding` and `parse_markdown` (ADR-0005). A single shared type means each adapter catches one exception, not a per-stack tuple.

### `retryable` is the only discriminator

The wrapper maps the three OpenAI exception groups onto `retryable`: `APITimeoutError`/`RateLimitError` → `retryable=True`; `AuthenticationError` and any other `APIError` → `retryable=False`. The precise `kind` (`openai_transient` / `openai_auth` / `openai_api`) stays in the `chat_error` log line (§5) — it is **not** carried on the exception, because no adapter renders an auth failure differently from a generic API error (neither the model nor a CLI user can self-fix an auth failure; the human-facing distinction already lives in `message`).

### Logging is unchanged; only the raised type changes

The wrapper still emits `chat_error` with the correct `kind=` tag **before** raising (§4.2's "every branch logs" requirement; §5.1 per-package channel). Only the final `raise HTTPException(...) from exc` becomes `raise LLMError(...) from exc`.

### Each adapter renders `LLMError`

| Adapter | Rendering |
|---|---|
| HTTP route (`routes.py`, both stacks) | `raise HTTPException(503 if e.retryable else 500, e.message) from e` — the §4.2 status table moves here |
| Gateway SSE generator | terminal `error{detail: e.message, retryable: e.retryable}` event (ADR-0009) |
| MCP (`kb_mcp`) | `isError` result, structured `{code: "LLM_UNAVAILABLE" if retryable else "LLM_ERROR", message}` |
| CLI (`kb_cli`) | non-zero exit code + `message` to stderr |

### CODING_STANDARD §4.2 is amended

§4.2's mandate changes from "the wrapper raises `HTTPException` per this table" to "the wrapper raises `LLMError`; the HTTP route maps it to status per this table." The status table itself is unchanged; it moves one layer out (wrapper → route).

## Considered Options

### Keep `HTTPException` in the wrapper; catch it in each non-HTTP adapter

Rejected. Every non-HTTP consumer would re-implement the same `except HTTPException` translation, the deep module stays coupled to FastAPI, and it directly contradicts the Phase 12 thesis that CLI/MCP are thin adapters over interface-agnostic deep modules. The appearance of a *second* non-HTTP consumer is precisely the signal that the coupling is wrong, and the right time to pay it down.

### A per-stack `LLMError` (each `retrieval.py` defines its own)

Rejected. The MCP adapter would have to `except (markdown_kb…LLMError, vector_rag…LLMError)`, and the type would be duplicated. The shared seam already exists (`vector_rag` imports `markdown_kb.grounding`), so a single shared type is both cleaner and consistent with the established dependency direction.

### A richer error carrying `kind` for granular surface codes

Rejected. Distinguishing `auth` from generic `api` at the interface gives the model or caller nothing actionable — neither is self-fixable — while `retryable` is the one bit every adapter actually branches on. `kind` granularity is for operator diagnostics and stays in the log.

## Consequences

- **§4.2 amended** (cross-reference this ADR). The LLM-call wrapper no longer imports `fastapi`; `retrieval.py` is decoupled from the HTTP transport.
- **Invariant**: `Cannot Confirm` is unaffected — it is a success path (§4.3, ADR-0001 / ADR-0004), never an `LLMError`. Empty/sub-threshold retrieval and a missing `.kb/index.json` resolve to Cannot Confirm with HTTP 200, not an error. There is therefore **no `INDEX_MISSING` error code** at any interface.
- Both stacks' `routes.py` gain an `except LLMError` clause; the gateway `_sse_generator` changes `except HTTPException` (deriving `retryable` from `status == 503`) to `except LLMError` reading `.retryable` directly.
- Tests asserting the wrapper raises `HTTPException` move to asserting `LLMError`; route-level tests continue to assert the HTTP status.
- New interfaces (MCP, CLI) render `LLMError` per their transport with no HTTP leak — the contract that makes Phase 12 a set of thin adapters over interface-agnostic deep modules.
