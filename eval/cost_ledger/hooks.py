"""Deep module per Ousterhout. Public surface: ``instrument_invoke``,
``record_usage_from_response``.

Hooks a production LLM-facing module's lazy-singleton getter (``get_llm``,
``get_verifier_llm``, ``get_ingest_llm``, ... — see ADR-0005's LLM-facing
surface enumeration) into a ``CostLedger``, so every plain ``.invoke()`` call
made through the client the wrapped getter returns is recorded automatically.
Mirrors CODING_STANDARD §2.7's own singleton pattern (tests swap the getter,
not the client) — here the eval harness swaps the getter to add
instrumentation instead of a stub.

Only the plain ``ChatOpenAI.invoke`` call shape is auto-recorded (the
answer-synthesis surfaces per ADR-0005's table: ``markdown_kb``/``vector_rag``
retrieval, ``hybrid_kb`` query). A ``with_structured_output`` chain (the
verifier / classifier / synthesis surfaces) does not expose ``usage_metadata``
on its parsed-object-only return unless the caller passes
``include_raw=True`` — changing that return shape is production behaviour
issue #657 does not touch (its AC 3 scopes the diff to the ledger plus the
ingest undercount fix). Call ``record_usage_from_response`` directly at those
call sites when the harness wires this ledger into the corpus v3 runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ledger import CostLedger
from .models import UsageMetadata


def _extract_usage_metadata(response: object) -> UsageMetadata:
    """Read ``usage_metadata`` off a LangChain response object (or a raw dict
    already shaped like one). Any other shape — including ``None`` — records
    a zero-token call: an LLM call still happened even when its usage is
    invisible at this seam."""
    if response is None:
        return UsageMetadata()
    raw = getattr(response, "usage_metadata", None)
    if raw is None and isinstance(response, dict):
        raw = response.get("usage_metadata")
    return UsageMetadata.from_raw(raw)


def record_usage_from_response(
    ledger: CostLedger, *, stack: str, phase: str, model: str, response: object
) -> None:
    """Record one call's usage, extracted from `response`, into `ledger`."""
    ledger.record(
        stack=stack, phase=phase, model=model, usage=_extract_usage_metadata(response)
    )


class _InvokeRecordingProxy:
    """Wraps one LLM client so every ``.invoke()`` call is recorded before
    returning. Every other attribute (including ``.with_structured_output``)
    delegates to the wrapped client unchanged — this proxy only observes the
    ``.invoke`` seam."""

    def __init__(
        self, llm: Any, ledger: CostLedger, *, stack: str, phase: str, model: str
    ) -> None:
        self._llm = llm
        self._ledger = ledger
        self._stack = stack
        self._phase = phase
        self._model = model

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        response = self._llm.invoke(*args, **kwargs)
        record_usage_from_response(
            self._ledger,
            stack=self._stack,
            phase=self._phase,
            model=self._model,
            response=response,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def instrument_invoke(
    getter: Callable[[], Any],
    ledger: CostLedger,
    *,
    stack: str,
    phase: str,
    model: str | None = None,
) -> Callable[[], Any]:
    """Wrap a lazy-singleton LLM getter so calling it returns a client whose
    ``.invoke()`` calls are recorded into `ledger` under (`stack`, `phase`).

    `model` overrides the recorded model name; when omitted it is read off
    the wrapped client's ``model_name`` attribute (``ChatOpenAI``'s
    convention), falling back to ``"unknown"``.
    """

    def wrapped_getter() -> Any:
        llm = getter()
        resolved_model = model or getattr(llm, "model_name", None) or "unknown"
        return _InvokeRecordingProxy(
            llm, ledger, stack=stack, phase=phase, model=resolved_model
        )

    return wrapped_getter
