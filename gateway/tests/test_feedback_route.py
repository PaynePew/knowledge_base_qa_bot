"""Gateway route tests: POST/GET /feedback — Reader Feedback (issue #558).

Acceptance criteria tested:
  AC1 — POST /feedback valid body -> 200; line appended to
        .kb/feedback.jsonl; `feedback` line in gateway/log.md.
  AC2 — Validation: comment >500 chars or unknown reaction -> 422;
        store >= 1MB -> 503 with the exact detail shape.
  AC3 — GET /feedback folds by answer_id last-wins (same answer_id x3
        records -> 1 folded row with the latest content) and returns
        correct counts.
  AC — /feedback is public: absent from both READ_PATHS and ADMIN_PATHS
       (middleware.py), so it is never budget/semaphore/token gated.

All tests are hermetic: FEEDBACK_PATH redirected to tmp_path, no
OPENAI_API_KEY, no @pytest.mark.live test (this route calls no LLM).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import gateway.app.feedback as feedback_module
from gateway.app.main import app as _gateway_app


@pytest.fixture(autouse=True)
def _redirect_feedback_path(tmp_path, monkeypatch):
    """Redirect FEEDBACK_PATH to tmp for write-isolation (CODING_STANDARD §6.5)."""
    monkeypatch.setattr(feedback_module, "FEEDBACK_PATH", tmp_path / ".kb" / "feedback.jsonl")


@pytest.fixture()
def client():
    return TestClient(_gateway_app)


def _valid_body(**overrides):
    body = {
        "answer_id": "answer-1",
        "reaction": "up",
        "query": "How long do refunds take?",
        "answer_preview": "Refunds take 5-7 business days.",
        "stack": "wiki",
        "session_id": "session-1",
        "citations": ["refund-policy#refund-policy"],
        "grounding": "claim_supported",
        "lang": "en",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# AC1: POST /feedback happy path — 200, appended line, log line
# ---------------------------------------------------------------------------


def test_post_feedback_valid_body_returns_200_with_id(client):
    resp = client.post("/feedback", json=_valid_body())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "id" in data and data["id"]


def test_post_feedback_appends_one_line_to_feedback_jsonl(client):
    client.post("/feedback", json=_valid_body())
    assert feedback_module.FEEDBACK_PATH.exists()
    lines = feedback_module.FEEDBACK_PATH.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["answer_id"] == "answer-1"
    assert rec["reaction"] == "up"
    assert rec["v"] == 1
    assert "id" in rec and "ts" in rec


def test_post_feedback_writes_feedback_log_line(client, monkeypatch):
    import gateway.app.logger as gw_logger

    log_path = feedback_module.FEEDBACK_PATH.parent / "log.md"
    monkeypatch.setattr(gw_logger, "LOG_PATH", log_path)

    client.post("/feedback", json=_valid_body())

    text = log_path.read_text(encoding="utf-8")
    assert "feedback |" in text
    assert "answer_id=answer-1" in text
    assert "reaction=up" in text


def test_post_feedback_comment_appends_a_new_record(client):
    client.post("/feedback", json=_valid_body())
    resp = client.post("/feedback", json=_valid_body(comment="Thanks, this helped!"))
    assert resp.status_code == 200, resp.text
    lines = feedback_module.FEEDBACK_PATH.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, "a comment submission appends a NEW record, not an in-place edit"


# ---------------------------------------------------------------------------
# AC2: validation — comment length, unknown reaction, store cap
# ---------------------------------------------------------------------------


def test_post_feedback_comment_over_500_chars_is_422(client):
    resp = client.post("/feedback", json=_valid_body(comment="x" * 501))
    assert resp.status_code == 422


def test_post_feedback_comment_at_500_chars_is_accepted(client):
    resp = client.post("/feedback", json=_valid_body(comment="x" * 500))
    assert resp.status_code == 200, resp.text


def test_post_feedback_unknown_reaction_is_422(client):
    resp = client.post("/feedback", json=_valid_body(reaction="sideways"))
    assert resp.status_code == 422


def test_post_feedback_missing_required_field_is_422(client):
    body = _valid_body()
    del body["query"]
    resp = client.post("/feedback", json=body)
    assert resp.status_code == 422


def test_post_feedback_store_full_returns_503_with_exact_detail(client, monkeypatch):
    """Store >= 1MB -> 503 {"detail": "feedback store full"} (AC2 exact shape)."""
    monkeypatch.setattr(feedback_module, "MAX_STORE_BYTES", 10)
    feedback_module.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    feedback_module.FEEDBACK_PATH.write_text("x" * 20, encoding="utf-8")

    resp = client.post("/feedback", json=_valid_body())
    assert resp.status_code == 503
    assert resp.json() == {"detail": "feedback store full"}


# ---------------------------------------------------------------------------
# AC3: GET /feedback — fold by answer_id (last write wins) + counts
# ---------------------------------------------------------------------------


def test_get_feedback_empty_store_returns_empty_shape(client):
    resp = client.get("/feedback")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"records": [], "counts": {"up": 0, "down": 0, "total_raw": 0}}


def test_get_feedback_folds_same_answer_id_to_one_record(client):
    client.post("/feedback", json=_valid_body(answer_id="a1", reaction="up"))
    client.post("/feedback", json=_valid_body(answer_id="a1", reaction="up", comment="great"))
    client.post(
        "/feedback",
        json=_valid_body(answer_id="a1", reaction="down", comment="actually no"),
    )

    resp = client.get("/feedback")
    data = resp.json()
    assert data["counts"]["total_raw"] == 3, "raw count is every line on disk"
    assert len(data["records"]) == 1, "3 records for the same answer_id fold to 1"
    folded = data["records"][0]
    assert folded["reaction"] == "down", "last write wins"
    assert folded["comment"] == "actually no", "last write wins"


def test_get_feedback_counts_up_down_over_folded_distinct_answers(client):
    # Two distinct answers, one resubmitted 3x (folds to 1) — counts must
    # reflect DISTINCT answers, not raw submission volume.
    client.post("/feedback", json=_valid_body(answer_id="a1", reaction="up"))
    client.post("/feedback", json=_valid_body(answer_id="a1", reaction="up", comment="x"))
    client.post("/feedback", json=_valid_body(answer_id="a2", reaction="down"))

    resp = client.get("/feedback")
    data = resp.json()
    assert data["counts"] == {"up": 1, "down": 1, "total_raw": 3}


def test_get_feedback_records_are_newest_first(client):
    client.post("/feedback", json=_valid_body(answer_id="a1"))
    client.post("/feedback", json=_valid_body(answer_id="a2"))
    client.post("/feedback", json=_valid_body(answer_id="a3"))

    resp = client.get("/feedback")
    ids = [r["answer_id"] for r in resp.json()["records"]]
    assert ids == ["a3", "a2", "a1"]


def test_get_feedback_caps_at_200_folded_records(client):
    for i in range(205):
        client.post("/feedback", json=_valid_body(answer_id=f"a{i}"))

    resp = client.get("/feedback")
    data = resp.json()
    assert len(data["records"]) == 200
    assert data["counts"]["total_raw"] == 205
    assert data["counts"]["up"] == 205, "counts reflect the FULL folded set, not the capped slice"


# ---------------------------------------------------------------------------
# Public surface: neither READ_PATHS nor ADMIN_PATHS (middleware.py)
# ---------------------------------------------------------------------------


def test_feedback_paths_are_not_read_gated():
    from gateway.app.middleware import READ_PATHS

    assert "/feedback" not in READ_PATHS


def test_feedback_paths_are_not_admin_gated():
    from gateway.app.middleware import ADMIN_PATHS

    assert "/feedback" not in ADMIN_PATHS


def test_post_feedback_admitted_even_with_admin_token_set(client, monkeypatch):
    """A public path is never token-gated, even when KB_ADMIN_TOKEN is set."""
    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")
    resp = client.post("/feedback", json=_valid_body())
    assert resp.status_code == 200, resp.text
