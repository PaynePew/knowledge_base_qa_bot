"""Trust-level path routing tests (CODING_STANDARD §6.6, issue #328).

Asserts that an offline ``--fake-embeddings`` run writes ONLY to the tracer
artifact path (``report.offline-tracer.md`` + ``charts-offline/``) and never
touches the canonical ``report.md`` / ``charts/`` paths.

Hermetic: monkeypatches the module-level path constants in ``runner`` so both
canonical and tracer outputs land in ``tmp_path`` (never on disk).  Uses the
same ``fake_vector_index`` fixture pattern as the rest of the eval suite (no
real API calls; ``embedding_mode="fake"`` is the offline path).
"""

from __future__ import annotations

from eval.paraphrase_comparison import runner
from eval.paraphrase_comparison.run_comparison import main


def test_offline_run_writes_only_tracer_not_canonical(
    tmp_path, fake_vector_index, monkeypatch
):
    """A --fake-embeddings run must not clobber the canonical report.md / charts/.

    AC#1 + AC#5 (issue #328): the offline tracer path receives all output;
    the canonical names stay absent; the tracer report's first line is the
    loud placeholder header constant (not an inline string).
    """
    canonical_report = tmp_path / "report.md"
    canonical_charts = tmp_path / "charts"
    tracer_report = tmp_path / "report.offline-tracer.md"
    tracer_charts = tmp_path / "charts-offline"

    # Redirect module constants to tmp so no real disk paths are exercised.
    monkeypatch.setattr(runner, "REPORT_PATH", canonical_report)
    monkeypatch.setattr(runner, "OFFLINE_TRACER_REPORT_PATH", tracer_report)
    monkeypatch.setattr(runner, "OFFLINE_CHARTS_DIR", tracer_charts)

    exit_code = main(["--fake-embeddings"])
    assert exit_code == 0

    # Tracer outputs must exist.
    assert tracer_report.exists(), "tracer report not written"
    assert tracer_charts.is_dir(), "tracer charts dir not created"
    assert list(tracer_charts.glob("*.png")), "no PNGs in charts-offline"

    # Canonical outputs must NOT exist — the footgun this slice prevents.
    assert not canonical_report.exists(), "fake run must NOT write canonical report.md"
    assert not canonical_charts.exists(), "fake run must NOT create canonical charts/"

    # Tracer report's very first line must be the loud header constant.
    content = tracer_report.read_text(encoding="utf-8")
    first_line = content.split("\n")[0]
    assert first_line == runner.OFFLINE_TRACER_HEADER, (
        f"expected first line to be OFFLINE_TRACER_HEADER, got: {first_line!r}"
    )
