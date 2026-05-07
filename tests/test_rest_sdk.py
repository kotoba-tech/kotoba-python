"""Unit tests for the REST SDK (``submit_job`` / ``get_job`` / ``transcribe``).

Uses the ``responses`` library to stub the HTTP round-trip so tests don't
depend on the real server being up. Run with: ``pytest tests/test_rest_sdk.py``.
"""

from __future__ import annotations

from io import BytesIO

import pytest
import responses

import kotoba
from kotoba.errors import (
    AuthError,
    JobNotFoundError,
    ProtocolError,
    TranscriptionError,
)

BASE_URL = "http://fake.example/v1"
JOBS_URL = f"{BASE_URL}/transcription_jobs"


@pytest.fixture
def fake_wav(tmp_path):
    path = tmp_path / "tiny.wav"
    # 44-byte WAV header + 2 bytes of pcm16 silence — content doesn't matter.
    path.write_bytes(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
                     b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00")
    return path


def _client() -> kotoba.KotobaClient:
    return kotoba.KotobaClient(api_key="token-abc", url=BASE_URL, max_retries=0)


# ---------- submit_job / get_job -----------------------------------------


@responses.activate
def test_submit_job_returns_job_id(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"job_id": "abc-123"},
        status=202,
    )
    job = _client().asr.submit_job(fake_wav, language="ja")
    assert job.job_id == "abc-123"
    # Also verify we forwarded Authorization.
    assert "Authorization" in responses.calls[0].request.headers
    assert responses.calls[0].request.headers["Authorization"] == "Bearer token-abc"


@responses.activate
def test_submit_job_sends_with_timestamps(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"job_id": "ts-1"},
        status=202,
    )
    _client().asr.submit_job(fake_wav, language="ja", with_timestamps=True)
    body = responses.calls[0].request.body
    # Multipart body contains the form fields as plain bytes.
    assert b"with_timestamps" in body
    assert b"true" in body


@responses.activate
def test_get_job_processing_returns_state():
    responses.add(
        responses.GET,
        f"{JOBS_URL}/abc",
        status=202,
        json={"detail": "still processing"},
    )
    status = _client().asr.get_job("abc")
    assert status.state == kotoba.JobState.processing


@responses.activate
def test_get_job_done_returns_transcription():
    responses.add(
        responses.GET,
        f"{JOBS_URL}/done-id",
        json={"state": "done", "transcription": "こんにちは"},
        status=200,
    )
    status = _client().asr.get_job("done-id")
    assert status.state == kotoba.JobState.done
    assert status.transcription == "こんにちは"


@responses.activate
def test_get_job_not_found_raises():
    responses.add(
        responses.GET,
        f"{JOBS_URL}/nope",
        json={"detail": "Job not found"},
        status=404,
    )
    with pytest.raises(JobNotFoundError):
        _client().asr.get_job("nope")


# ---------- transcribe (POST + poll) -------------------------------------


@responses.activate
def test_transcribe_polls_until_done(fake_wav):
    responses.add(responses.POST, JOBS_URL, json={"job_id": "id-1"}, status=202)
    responses.add(
        responses.GET,
        f"{JOBS_URL}/id-1",
        json={"detail": "still processing"},
        status=202,
    )
    responses.add(
        responses.GET,
        f"{JOBS_URL}/id-1",
        json={"state": "done", "transcription": "ok"},
        status=200,
    )
    result = _client().asr.transcribe(
        fake_wav,
        language="ja",
        poll_interval=0.01,
        poll_backoff=1.0,
        max_poll_interval=0.01,
        timeout=5.0,
    )
    assert result.text == "ok"
    assert result.job_id == "id-1"


@responses.activate
def test_transcribe_error_state_raises(fake_wav):
    responses.add(responses.POST, JOBS_URL, json={"job_id": "id-2"}, status=202)
    responses.add(
        responses.GET,
        f"{JOBS_URL}/id-2",
        json={"state": "error", "error_message": "worker blew up"},
        status=200,
    )
    with pytest.raises(TranscriptionError) as exc:
        _client().asr.transcribe(fake_wav, poll_interval=0.01, timeout=2.0)
    assert "worker blew up" in str(exc.value)


@responses.activate
def test_transcribe_timeout(fake_wav):
    responses.add(responses.POST, JOBS_URL, json={"job_id": "id-3"}, status=202)
    responses.add(
        responses.GET,
        f"{JOBS_URL}/id-3",
        json={"detail": "still processing"},
        status=202,
    )
    with pytest.raises(kotoba.TimeoutError):
        _client().asr.transcribe(
            fake_wav,
            poll_interval=0.01,
            poll_backoff=1.0,
            max_poll_interval=0.01,
            timeout=0.05,
        )


# ---------- auth / protocol --------------------------------------------


@responses.activate
def test_auth_error_on_401(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"detail": "Unauthorized"},
        status=401,
    )
    with pytest.raises(AuthError):
        _client().asr.submit_job(fake_wav)


@responses.activate
def test_protocol_error_on_400(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"detail": "empty audio payload"},
        status=400,
    )
    with pytest.raises(ProtocolError):
        _client().asr.submit_job(fake_wav)


@responses.activate
def test_auth_error_unwraps_openai_envelope(fake_wav):
    """OpenAI-style {"error": {"message": "..."}} payloads should surface the
    inner message string, not the raw dict, in the exception."""

    responses.add(
        responses.POST,
        JOBS_URL,
        json={
            "error": {
                "code": "invalid_api_key",
                "type": "invalid_request_error",
                "message": "Not Authorized. Check your API key at https://dashboard.kotobatech.ai/",
                "param": None,
            }
        },
        status=401,
    )
    with pytest.raises(AuthError) as exc:
        _client().asr.submit_job(fake_wav)
    assert (
        str(exc.value)
        == "Not Authorized. Check your API key at https://dashboard.kotobatech.ai/"
    )


@responses.activate
def test_protocol_error_preserves_fastapi_validation_detail(fake_wav):
    """FastAPI 422 returns ``detail`` as a list of validation entries.
    The exception message must surface the structured info (stringified),
    not collapse to a generic ``HTTP 422 from ...`` placeholder."""

    detail = [
        {
            "loc": ["body", "language"],
            "msg": "field required",
            "type": "value_error.missing",
        }
    ]
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"detail": detail},
        status=422,
    )
    with pytest.raises(ProtocolError) as exc:
        _client().asr.submit_job(fake_wav)
    assert "field required" in str(exc.value)
    assert exc.value.payload == {"detail": detail}


# ---------- URL / env handling -----------------------------------------


def test_client_requires_url(monkeypatch):
    # Ensure neither kwarg nor env supplies a URL.
    for var in ("KOTOBA_API_KEY", "KOTOBA_ASR_REST_URL"):
        monkeypatch.delenv(var, raising=False)

    client = kotoba.KotobaClient(api_key="x")  # no url -> REST disabled

    with pytest.raises(RuntimeError):
        client.asr.submit_job(BytesIO(b"dummy"))  # type: ignore[arg-type]


def test_client_url_from_env(monkeypatch):
    monkeypatch.setenv("KOTOBA_ASR_REST_URL", BASE_URL)
    client = kotoba.KotobaClient(api_key="x")
    assert client.url == BASE_URL
