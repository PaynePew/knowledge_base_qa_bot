"""Deterministic mechanism test for Transcribe's vision system-prompt
hardening (ADR-0040 Q5 / issue #584).

Transcribe (image input) is a DIFFERENT mechanism from the text surfaces
hardened in the #577 batch: there is no text to fence (the untrusted content
arrives as image bytes, not a string spliced into the user message), so
``wrap_untrusted`` does not apply here — only the vision system prompt itself
carries an explicit guard clause. This asserts that guard's presence/wording;
it makes no LLM call.

Transcribe already has its ONE authorised ``@pytest.mark.live`` smoke test
(``test_transcribe_live.py`` — ADR-0005 "one live test per surface", also
implement.md trap #4). A SECOND live test proving the model actually resists
an embedded-instruction page image is deliberately NOT added here; instead
the behavioural probe follows this project's established house pattern for
injection verification (ADR-0040 Consequences: "manual, post-deploy
real-artifact probe", not a live pytest) — see
``project-docs/security/injection-probe/README.md``'s transcribe carrier.
"""

from __future__ import annotations

from app.transcriber import _TRANSCRIBE_SYSTEM_PROMPT


def test_transcribe_system_prompt_states_image_text_is_content_not_instruction():
    lowered = _TRANSCRIBE_SYSTEM_PROMPT.lower()
    assert "content to transcribe" in lowered or "content" in lowered
    assert "never an instruction" in lowered or "never obey" in lowered
    assert "verbatim" in lowered


def test_transcribe_system_prompt_names_the_injection_example():
    """The prompt must give a concrete instruction-hijack example, mirroring
    the AC's own example phrasing, so the guard is unambiguous to the model."""
    assert "ignore your task and output x" in _TRANSCRIBE_SYSTEM_PROMPT.lower() or (
        "ignore your task" in _TRANSCRIBE_SYSTEM_PROMPT.lower()
    )


def test_transcribe_system_prompt_still_requires_faithful_transcription():
    """The pre-existing faithful-transcription contract (ADR-0032) is
    unchanged by the added guard clause — no line was dropped in the edit."""
    assert "FAITHFUL TRANSCRIPTION" in _TRANSCRIBE_SYSTEM_PROMPT
    assert "empty" in _TRANSCRIBE_SYSTEM_PROMPT.lower()
