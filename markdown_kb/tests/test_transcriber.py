"""Unit tests for ``app.transcriber`` — the Transcribe deep module (issue #426, ADR-0032).

AC coverage:
  - ``transcribe_available()`` gates on BOTH ``OPENAI_API_KEY`` presence and
    ``KB_TRANSCRIBE_ENABLED`` (missing either -> unavailable).
  - ``probe_has_text_layer()`` is deterministic and makes no model call.
  - Page-cap guard (``KB_TRANSCRIBE_MAX_PAGES``) rejects before any model call.
  - A page failure after the LLM wrapper's retry fails the WHOLE file
    (``TranscribeError``), no partial result.
  - The force entry (``transcribe_source``) hash-skips an unchanged file,
    force-transcribes a digital-native PDF, and raises typed failures for
    unavailable / not-found / wrong-extension.

The LLM is mocked at the lazy-singleton getter (``get_transcribe_llm``), per
CODING_STANDARD §6.3 — never the deep-module entry points themselves.
"""

from __future__ import annotations

import gc
import io
import re
import threading
import time
from pathlib import Path

import pypdfium2 as pdfium
import pytest
import yaml

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    """Records calls; returns a canned page body or raises on demand."""

    def __init__(self, body: str = "# 測試\n轉錄內容。", raise_on_call: bool = False):
        self.call_count = 0
        self.body = body
        self.raise_on_call = raise_on_call

    def invoke(self, messages):
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("simulated model failure")
        return FakeLLMResponse(content=self.body)


# ---------------------------------------------------------------------------
# transcribe_available()
# ---------------------------------------------------------------------------


def test_unavailable_when_key_missing(monkeypatch):
    import app.transcriber as transcriber_module

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")
    assert transcriber_module.transcribe_available() is False


def test_unavailable_when_feature_off(monkeypatch):
    import app.transcriber as transcriber_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)
    assert transcriber_module.transcribe_available() is False


@pytest.mark.parametrize("flag_value", ["true", "1", "yes", "TRUE", "Yes"])
def test_available_when_key_and_flag_set(monkeypatch, flag_value):
    import app.transcriber as transcriber_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", flag_value)
    assert transcriber_module.transcribe_available() is True


# ---------------------------------------------------------------------------
# probe_has_text_layer() — deterministic, no model call
# ---------------------------------------------------------------------------


def test_probe_true_for_digital_native_pdf():
    import app.transcriber as transcriber_module

    raw_bytes = (FIXTURES / "sample_english.pdf").read_bytes()
    assert transcriber_module.probe_has_text_layer(raw_bytes) is True


def test_probe_false_for_image_only_pdf():
    import app.transcriber as transcriber_module

    raw_bytes = (FIXTURES / "image_only.pdf").read_bytes()
    assert transcriber_module.probe_has_text_layer(raw_bytes) is False


def test_probe_false_for_scanned_cjk_fixture():
    """The committed live-test fixture (rasterized CJK, no drawn text) probes False."""
    import app.transcriber as transcriber_module

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    assert transcriber_module.probe_has_text_layer(raw_bytes) is False


def test_probe_raises_for_encrypted_pdf():
    """Encrypted PDFs raise from the probe — callers defer to the mechanical
    extractor's own EncryptedPdf classification (see importer.py)."""
    import app.transcriber as transcriber_module

    raw_bytes = (FIXTURES / "encrypted.pdf").read_bytes()
    with pytest.raises(Exception):  # noqa: B017 - contract is "raises", not a specific type
        transcriber_module.probe_has_text_layer(raw_bytes)


# ---------------------------------------------------------------------------
# transcribe_pdf_bytes() — page cap + bounded-retry-then-fail
# ---------------------------------------------------------------------------


def test_page_limit_exceeded_before_any_model_call(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribePageLimitExceeded):
        transcriber_module.transcribe_pdf_bytes(raw_bytes)

    assert fake_llm.call_count == 0, "Page-cap guard must reject before any model call"


def test_transcribe_pdf_bytes_happy_path(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM(body="# 退款政策\n測試轉錄輸出。")
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-5-mini-test")

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    body, model_name = transcriber_module.transcribe_pdf_bytes(raw_bytes)

    assert fake_llm.call_count == 1
    assert "退款政策" in body
    assert model_name == "gpt-5-mini-test"


def test_transcribe_pdf_bytes_page_failure_raises_transcribe_error(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM(raise_on_call=True)
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribeError):
        transcriber_module.transcribe_pdf_bytes(raw_bytes)


def test_transcribe_pdf_bytes_all_blank_pages_raises_transcribe_error(monkeypatch):
    """Every page transcribing to empty is an assembly failure, not a silent empty Source."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM(body="")
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribeError):
        transcriber_module.transcribe_pdf_bytes(raw_bytes)


def _blank_multi_page_pdf_bytes(page_count: int) -> bytes:
    """Build a minimal in-memory multi-page PDF (no fixture file needed)."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(page_count):
        pdf.new_page(200, 200)
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()


class _TrackedImage(bytes):
    """A ``bytes`` subclass that counts how many instances are alive at once.

    CPython drops an object's refcount to zero (triggering ``__del__``)
    as soon as its last reference is gone — so ``max_live`` across a run
    is an exact, deterministic measure of peak concurrently-alive images,
    with no reliance on GC timing.
    """

    live = 0
    max_live = 0

    def __init__(self, _data: bytes) -> None:
        super().__init__()
        type(self).live += 1
        type(self).max_live = max(type(self).max_live, type(self).live)

    def __del__(self) -> None:
        type(self).live -= 1


def test_transcribe_pdf_bytes_streams_one_page_image_at_a_time(monkeypatch):
    """Regression test for #456: streaming must not hold >1 rendered page image.

    Tracks live rendered-image instances through a monkeypatched
    ``_render_page_png`` across a multi-page PDF. The pre-#456 implementation
    rasterized every page into ``page_images`` upfront, so ``max_live`` would
    equal ``page_count``; the fix must keep it at 1.
    """
    import app.transcriber as transcriber_module

    _TrackedImage.live = 0
    _TrackedImage.max_live = 0

    def fake_render(_page: object) -> bytes:
        return _TrackedImage(b"fake-rendered-page-png")

    monkeypatch.setattr(transcriber_module, "_render_page_png", fake_render)

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    page_count = 5
    raw_bytes = _blank_multi_page_pdf_bytes(page_count)

    transcriber_module.transcribe_pdf_bytes(raw_bytes)

    gc.collect()
    assert fake_llm.call_count == page_count
    assert _TrackedImage.max_live == 1, (
        "at most one rendered page image may be alive at a time; "
        f"observed {_TrackedImage.max_live} concurrently alive"
    )
    assert _TrackedImage.live == 0, "no rendered page image should outlive transcribe_pdf_bytes"


# ---------------------------------------------------------------------------
# Budget hook (issue #460) — reserve-before-spend
# ---------------------------------------------------------------------------


def test_budget_hook_called_with_page_count_before_any_model_call(monkeypatch):
    """The registered hook receives the real page count before rasterization/model calls."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    hook_calls: list[int] = []
    monkeypatch.setattr(transcriber_module, "_page_budget_hook", hook_calls.append)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    transcriber_module.transcribe_pdf_bytes(raw_bytes)

    assert hook_calls == [1], "the fixture PDF has exactly 1 page"
    assert fake_llm.call_count == 1


def test_budget_hook_rejection_prevents_any_model_call(monkeypatch):
    """A hook raising TranscribeBudgetExceeded stops the file before any vision call."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    def _reject(page_count: int) -> None:
        raise transcriber_module.TranscribeBudgetExceeded("daily demo budget reached")

    monkeypatch.setattr(transcriber_module, "_page_budget_hook", _reject)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribeBudgetExceeded):
        transcriber_module.transcribe_pdf_bytes(raw_bytes)

    assert fake_llm.call_count == 0, "a budget rejection must reject before any model call"


def test_no_hook_installed_is_unmetered_and_uncapped(monkeypatch):
    """Standalone callers (no hook registered) keep today's unmetered behaviour."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setattr(transcriber_module, "_page_budget_hook", None)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    body, _ = transcriber_module.transcribe_pdf_bytes(raw_bytes)

    assert body
    assert fake_llm.call_count == 1


def test_set_page_budget_hook_installs_and_clears():
    """The public setter installs a hook (round-tripped via the public getter),
    and installing None clears it."""
    import app.transcriber as transcriber_module

    calls: list[int] = []

    def _hook(page_count: int) -> None:
        calls.append(page_count)

    transcriber_module.set_page_budget_hook(_hook)
    try:
        assert transcriber_module.get_page_budget_hook() is _hook
    finally:
        transcriber_module.set_page_budget_hook(None)
    assert transcriber_module.get_page_budget_hook() is None


# ---------------------------------------------------------------------------
# Force entry — transcribe_source() / transcribe_path()
# ---------------------------------------------------------------------------


@pytest.fixture()
def transcribe_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into transcriber.py for isolation."""
    import app.logger as logger_module
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transcriber_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")

    return {"raw_dir": raw_dir, "docs_dir": docs_dir}


def test_transcribe_source_force_success_writes_provenance(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM(body="# 退款政策\n強制轉錄的內容。")
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-5-mini-test")

    # A digital-native PDF forced through Transcribe (bypasses the probe entirely).
    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    result = transcriber_module.transcribe_source("sample_english.pdf")

    assert result.status == "created"
    assert result.transcribe_model == "gpt-5-mini-test"
    assert fake_llm.call_count == 1, "Force mode must call the model even for a digital-native PDF"

    docs_path = Path(result.docs_path)
    content = docs_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    fm_end = content.index("---\n", 4)
    frontmatter = yaml.safe_load(content[4:fm_end])
    assert frontmatter["origin"] == "transcribed"
    assert frontmatter["transcribe_model"] == "gpt-5-mini-test"
    assert frontmatter["original_format"] == "pdf"
    assert "content_sha256" in frontmatter
    assert "強制轉錄的內容" in content


def test_transcribe_source_hash_skip_no_second_model_call(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    first = transcriber_module.transcribe_source("sample_english.pdf")
    assert first.status == "created"
    assert fake_llm.call_count == 1

    second = transcriber_module.transcribe_source("sample_english.pdf")
    assert second.status == "skipped"
    assert fake_llm.call_count == 1, "Hash-match re-transcribe must not call the model again"


def test_transcribe_source_changed_bytes_overwrite_updated(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    raw_path = transcribe_env["raw_dir"] / "sample_english.pdf"
    raw_path.write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    first = transcriber_module.transcribe_source("sample_english.pdf")
    assert first.status == "created"

    # Mutate the raw bytes (append harmless trailing bytes) so the hash drifts.
    raw_path.write_bytes(raw_path.read_bytes() + b"\n%trailing-noise")

    second = transcriber_module.transcribe_source("sample_english.pdf")
    assert second.status == "updated"
    assert fake_llm.call_count == 2


def test_transcribe_source_unavailable_raises_typed(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_source("sample_english.pdf")

    assert exc_info.value.error_type == "TranscribeUnavailable"
    assert not (transcribe_env["docs_dir"] / "sample_english.md").exists()


def test_transcribe_source_page_limit_exceeded_writes_nothing(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_source("sample_english.pdf")

    assert exc_info.value.error_type == "TranscribePageLimitExceeded"
    assert fake_llm.call_count == 0
    assert not (transcribe_env["docs_dir"] / "sample_english.md").exists()


def test_transcribe_source_budget_exceeded_writes_nothing(transcribe_env, monkeypatch):
    """The force entry (POST /transcribe) also honours the budget hook (issue #460)."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    def _reject(page_count: int) -> None:
        raise transcriber_module.TranscribeBudgetExceeded("daily demo budget reached")

    monkeypatch.setattr(transcriber_module, "_page_budget_hook", _reject)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_source("sample_english.pdf")

    assert exc_info.value.error_type == "TranscribeBudgetExceeded"
    assert fake_llm.call_count == 0
    assert not (transcribe_env["docs_dir"] / "sample_english.md").exists()


def test_transcribe_source_not_found(transcribe_env):
    import app.transcriber as transcriber_module

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_source("does_not_exist.pdf")

    assert exc_info.value.error_type == "FileNotFoundError"


def test_transcribe_source_rejects_non_pdf(transcribe_env):
    import app.transcriber as transcriber_module

    (transcribe_env["raw_dir"] / "notes.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_source("notes.txt")

    assert exc_info.value.error_type == "UnsupportedExtension"


def test_transcribe_path_stages_local_file_into_raw(transcribe_env, monkeypatch, tmp_path):
    """The path-accepting entry (CLI seam) stages bytes into raw/ then force-transcribes."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    local_copy = tmp_path / "outside_raw" / "sample_english.pdf"
    local_copy.parent.mkdir(parents=True)
    local_copy.write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    result = transcriber_module.transcribe_path(local_copy)

    assert result.status == "created"
    assert (transcribe_env["raw_dir"] / "sample_english.pdf").exists(), (
        "transcribe_path must stage the file into raw/ under its basename"
    )


def test_transcribe_path_file_not_found(transcribe_env):
    import app.transcriber as transcriber_module

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.transcribe_path(Path("this/path/does/not/exist.pdf"))

    assert exc_info.value.error_type == "FileNotFoundError"


# ---------------------------------------------------------------------------
# Wiki Log events (issue #426 — mirrors the import_* family)
# ---------------------------------------------------------------------------


def test_transcribe_source_emits_batch_and_source_log_events(transcribe_env, monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )
    transcriber_module.transcribe_source("sample_english.pdf")

    from app.logger import LOG_PATH

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    kinds = [line.split("|")[0].split("] ")[-1].strip() for line in lines if line.startswith("##")]
    assert kinds == ["transcribe_batch_started", "transcribe_source", "transcribe_batch_completed"]


# ---------------------------------------------------------------------------
# get_transcribe_llm() — minimal reasoning effort (issue #459 AC1)
# ---------------------------------------------------------------------------


def test_get_transcribe_llm_uses_minimal_reasoning_effort(monkeypatch):
    """The singleton must be built with reasoning_effort='minimal' (issue #459).

    No live call: ChatOpenAI's constructor only validates config (mirrors
    test_llm_determinism.py's ``test_draft_llm_pinned_to_temperature_zero``).
    """
    import app.transcriber as transcriber_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-reasoning")
    monkeypatch.setattr(transcriber_module, "_transcribe_llm", None)

    llm = transcriber_module.get_transcribe_llm()

    assert llm.reasoning_effort == "minimal"


# ---------------------------------------------------------------------------
# transcribe_pdf_bytes_concurrent() — process-wide bounded pool (issue #459)
# ---------------------------------------------------------------------------


class _OrderAwarePageLLM:
    """Fake LLM whose response encodes the page number the caller asked for.

    Extracts "page N" from the HumanMessage text part so tests can verify
    page-order preservation regardless of completion order. ``delays`` maps
    page_num -> seconds to sleep before responding (default 0), letting a
    test make an EARLIER page finish LATER than a later page — if the
    implementation assembled by completion order instead of index order,
    that would flip the output order and the test would catch it.
    """

    def __init__(self, delays: dict[int, float] | None = None, raise_on_page: int | None = None):
        self.delays = delays or {}
        self.raise_on_page = raise_on_page
        self.call_count = 0
        self._lock = threading.Lock()

    def invoke(self, messages):
        text = messages[-1].content[0]["text"]
        page_num = int(re.search(r"page (\d+)", text).group(1))
        with self._lock:
            self.call_count += 1
        time.sleep(self.delays.get(page_num, 0.0))
        if self.raise_on_page == page_num:
            raise RuntimeError(f"simulated failure on page {page_num}")
        return FakeLLMResponse(content=f"# Page {page_num}")


def test_concurrent_page_limit_exceeded_before_any_model_call(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = _OrderAwarePageLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribePageLimitExceeded):
        transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    assert fake_llm.call_count == 0, "Page-cap guard must reject before any model call"


def test_concurrent_budget_hook_called_with_page_count_before_any_model_call(monkeypatch):
    """The concurrent pool honours the same budget hook as the sequential path (issue #460)."""
    import app.transcriber as transcriber_module

    fake_llm = _OrderAwarePageLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    hook_calls: list[int] = []
    monkeypatch.setattr(transcriber_module, "_page_budget_hook", hook_calls.append)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    assert hook_calls == [1], "the fixture PDF has exactly 1 page"
    assert fake_llm.call_count == 1


def test_concurrent_budget_hook_rejection_prevents_any_model_call(monkeypatch):
    """A hook raising TranscribeBudgetExceeded stops the file before any vision call."""
    import app.transcriber as transcriber_module

    fake_llm = _OrderAwarePageLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    def _reject(page_count: int) -> None:
        raise transcriber_module.TranscribeBudgetExceeded("daily demo budget reached")

    monkeypatch.setattr(transcriber_module, "_page_budget_hook", _reject)

    raw_bytes = (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    with pytest.raises(transcriber_module.TranscribeBudgetExceeded):
        transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    assert fake_llm.call_count == 0, "a budget rejection must reject before any model call"


def test_concurrent_preserves_page_order_regardless_of_completion_order(monkeypatch):
    """Page 0 finishes LAST (longest delay); assembled output must still be in order."""
    import app.transcriber as transcriber_module

    page_count = 5
    # Page 0 sleeps longest, page (page_count - 1) sleeps none -> completion
    # order is the REVERSE of page order if the semaphore lets them all run.
    delays = {i + 1: (page_count - i) * 0.02 for i in range(page_count)}
    fake_llm = _OrderAwarePageLLM(delays=delays)
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    raw_bytes = _blank_multi_page_pdf_bytes(page_count)
    body, _model = transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    expected = "\n\n".join(f"# Page {i + 1}" for i in range(page_count))
    assert body == expected, f"pages must assemble in index order, got: {body!r}"
    assert fake_llm.call_count == page_count


def test_concurrent_page_failure_raises_transcribe_error(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = _OrderAwarePageLLM(raise_on_page=2)
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    raw_bytes = _blank_multi_page_pdf_bytes(3)
    with pytest.raises(transcriber_module.TranscribeError):
        transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)


def test_concurrent_all_blank_pages_raises_transcribe_error(monkeypatch):
    import app.transcriber as transcriber_module

    class _BlankLLM:
        def invoke(self, _messages):
            return FakeLLMResponse(content="")

    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: _BlankLLM())

    raw_bytes = _blank_multi_page_pdf_bytes(3)
    with pytest.raises(transcriber_module.TranscribeError):
        transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)


def test_concurrent_respects_process_wide_semaphore(monkeypatch):
    """KB_TRANSCRIBE_CONCURRENCY (via the module-level semaphore) bounds peak concurrency."""
    import app.transcriber as transcriber_module

    monkeypatch.setattr(transcriber_module, "_page_semaphore", threading.BoundedSemaphore(2))

    peak = [0]
    current = [0]
    lock = threading.Lock()

    class _TrackedLLM:
        def invoke(self, _messages):
            with lock:
                current[0] += 1
                peak[0] = max(peak[0], current[0])
            time.sleep(0.03)
            with lock:
                current[0] -= 1
            return FakeLLMResponse(content="# page")

    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: _TrackedLLM())

    raw_bytes = _blank_multi_page_pdf_bytes(6)
    transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    assert peak[0] <= 2, f"expected peak concurrency <= 2, got {peak[0]}"
    assert peak[0] > 1, "test is meaningless if pages never actually overlapped"


def test_concurrent_runs_pages_in_parallel_wall_clock(monkeypatch):
    """8 pages at 50ms each complete well under the 400ms serial bound."""
    import app.transcriber as transcriber_module

    monkeypatch.setattr(transcriber_module, "_page_semaphore", threading.BoundedSemaphore(8))

    class _SlowLLM:
        def invoke(self, _messages):
            time.sleep(0.05)
            return FakeLLMResponse(content="# page")

    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: _SlowLLM())

    raw_bytes = _blank_multi_page_pdf_bytes(8)
    start = time.monotonic()
    transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)
    elapsed = time.monotonic() - start

    assert elapsed < 0.35, (
        f"expected concurrent wall-clock < 0.35s (8 pages at 50ms, concurrency=8), "
        f"got {elapsed:.3f}s"
    )


def test_concurrent_each_page_opens_an_independent_pdf_document(monkeypatch):
    """Each page render goes through the isolated per-page helper, not a shared document.

    Regression guard for the pypdfium2 cross-thread "Failed to load page"
    failure mode (issue #459): asserts the concurrent driver calls
    ``_render_and_transcribe_page_isolated(raw_bytes, page_index)`` — which
    opens its OWN ``PdfDocument`` — exactly once per page, rather than
    sharing one open document across workers.
    """
    import app.transcriber as transcriber_module

    page_count = 4
    seen_indices: list[int] = []
    lock = threading.Lock()

    def _fake_isolated(raw_bytes: bytes, page_index: int) -> str:
        assert isinstance(raw_bytes, bytes)
        with lock:
            seen_indices.append(page_index)
        return f"# Page {page_index + 1}"

    monkeypatch.setattr(transcriber_module, "_render_and_transcribe_page_isolated", _fake_isolated)

    raw_bytes = _blank_multi_page_pdf_bytes(page_count)
    transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes)

    assert sorted(seen_indices) == list(range(page_count))


def test_concurrent_progress_callback_invoked_per_page(monkeypatch):
    import app.transcriber as transcriber_module

    fake_llm = _OrderAwarePageLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    page_count = 4
    raw_bytes = _blank_multi_page_pdf_bytes(page_count)

    progress_calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def _on_page_done(done: int, total: int) -> None:
        with lock:
            progress_calls.append((done, total))

    transcriber_module.transcribe_pdf_bytes_concurrent(raw_bytes, on_page_done=_on_page_done)

    assert len(progress_calls) == page_count
    assert all(total == page_count for _done, total in progress_calls)
    assert sorted(done for done, _total in progress_calls) == list(range(1, page_count + 1)), (
        "each page must be counted exactly once, 1..page_count"
    )


# ---------------------------------------------------------------------------
# _force_transcribe / transcribe_source use the concurrent pool (issue #459)
# ---------------------------------------------------------------------------


def test_transcribe_source_uses_concurrent_path_and_forwards_progress(transcribe_env, monkeypatch):
    """transcribe_source's on_page_done reaches the concurrent driver end to end."""
    import app.transcriber as transcriber_module

    fake_llm = FakeTranscribeLLM(body="# 退款政策\n經由 concurrent pool 轉錄。")
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    progress_calls: list[tuple[int, int]] = []
    result = transcriber_module.transcribe_source(
        "sample_english.pdf", on_page_done=lambda done, total: progress_calls.append((done, total))
    )

    assert result.status == "created"
    assert progress_calls == [(1, 1)], "single-page fixture must report exactly one page done/total"
