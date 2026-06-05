"""Transport-agnostic domain exceptions (ADR-0015).

Defines the shared error types that cross the LLM-call boundary without
leaking any HTTP / transport concerns into deep modules.

``LLMError`` is defined once here in ``markdown_kb`` and imported by
``vector_rag`` — the same cross-package seam used for ``grounding`` and
``parse_markdown`` (ADR-0005).
"""

from __future__ import annotations


class LLMError(Exception):
    """Transport-agnostic LLM-call failure.  Interface adapters render it.

    Raised by ``_call_llm_with_error_handling`` in both stacks' retrieval
    modules instead of ``fastapi.HTTPException``.  Each interface adapter
    (HTTP routes, Gateway SSE generator, future MCP / CLI adapters) catches
    this exception and maps it to the appropriate transport representation.

    Attributes:
        retryable: ``True`` when the failure is transient (timeout,
            rate-limit) and the caller *should* retry.  ``False`` for auth
            failures, bad-API-key, or unexpected API errors.
        message: Human-readable description of the failure, safe to surface
            to the caller (no secrets, no stack traces).

    The precise ``kind`` (``openai_transient`` / ``openai_auth`` /
    ``openai_api``) is logged in the ``chat_error`` line before raising and
    is **not** carried on the exception — no interface renders auth
    differently from a generic API error at the UX level; the log is the
    operator-facing diagnostic channel (ADR-0015 §Decision).
    """

    def __init__(self, *, retryable: bool, message: str) -> None:
        self.retryable = retryable
        self.message = message
        super().__init__(message)
