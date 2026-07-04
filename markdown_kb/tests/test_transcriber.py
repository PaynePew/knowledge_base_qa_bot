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

from pathlib import Path

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
