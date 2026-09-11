"""Unit tests for the REST SDK (``submit_job`` / ``get_job`` / ``transcribe``).

Uses the ``responses`` library to stub the HTTP round-trip so tests don't
depend on the real server being up. Run with: ``pytest tests/test_rest_sdk.py``.
"""

from __future__ import annotations

from io import BytesIO

import httpx
import kotoba
import pytest
import requests
import responses
from kotoba._http import AsyncHttpSession
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
def test_submit_job_sends_kana(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"job_id": "kana-1"},
        status=202,
    )
    _client().asr.submit_job(
        fake_wav, language="ja", style_preference={"human_name": "kana"}
    )
    body = responses.calls[0].request.body
    # style_preference maps to the server's flat `human_name=kana` form field.
    assert b'name="human_name"\r\n\r\nkana\r\n' in body


@responses.activate
def test_submit_job_omits_kana_by_default(fake_wav):
    responses.add(
        responses.POST,
        JOBS_URL,
        json={"job_id": "no-kana"},
        status=202,
    )
    _client().asr.submit_job(fake_wav, language="ja")
    body = responses.calls[0].request.body
    # No style field is sent when kana is not requested (server defaults to "default").
    assert b"human_name" not in body


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
        fake_wav, batch=True,
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
        _client().asr.transcribe(fake_wav, batch=True, poll_interval=0.01, timeout=2.0)
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
            fake_wav, batch=True,
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


# ---------- fal.run endpoints: auth scheme + key source --------------------

FAL_BASE_URL = "https://fal.run/team/app"


@responses.activate
def test_fal_url_uses_key_auth_scheme(fake_wav):
    responses.add(
        responses.POST,
        f"{FAL_BASE_URL}/transcription_jobs",
        json={"job_id": "fal-1"},
        status=202,
    )
    client = kotoba.KotobaClient(api_key="fal-secret", url=FAL_BASE_URL, max_retries=0)
    client.asr.submit_job(fake_wav, language="ja")
    assert responses.calls[0].request.headers["Authorization"] == "Key fal-secret"



# ---------- transcribe(): one-shot mode (POST /v1/speech-to-text) -----------

STT_URL = f"{BASE_URL}/speech-to-text"  # BASE_URL already ends in /v1
STT_RESPONSE = {
    "language_code": "ja",
    "language_probability": 1.0,
    "text": "こんにちは",
    "words": [],
    "audio_duration_secs": 1.2,
}


def _part(body: bytes, name: str) -> list[bytes]:
    marker = f'name="{name}"'.encode()
    return [chunk for chunk in body.split(b"\r\n--") if marker in chunk]


@responses.activate
def test_transcribe_one_shot_request_shape(fake_wav):
    responses.add(responses.POST, STT_URL, json=STT_RESPONSE, status=200)
    result = _client().asr.transcribe(
        fake_wav,
        language="ja",
        keywords=["kotoba", "字幕"],
        style_preference={"human_name": "kana"},
        with_timestamps=True,
    )
    assert result.text == "こんにちは"
    assert len(responses.calls) == 1  # one request, no polling
    req = responses.calls[0].request
    assert req.headers["Authorization"] == "Bearer token-abc"
    assert "xi-api-key" not in req.headers  # removed server-side (#1593)
    body = req.body
    assert len(_part(body, "file")) == 1 and b"tiny.wav" in body
    assert b'name="file_format"\r\n\r\nother' in body
    assert b'name="language"\r\n\r\nja' in body
    assert len(_part(body, "keywords")) == 2  # repeated form fields
    assert b'name="kana"\r\n\r\ntrue' in body
    assert b'name="with_timestamps"\r\n\r\ntrue' in body


@responses.activate
def test_transcribe_one_shot_defaults(fake_wav):
    responses.add(responses.POST, STT_URL, json=STT_RESPONSE, status=200)
    result = _client().asr.transcribe(fake_wav)
    body = responses.calls[0].request.body
    assert b'name="language"' not in body  # server default applies
    assert b'name="kana"' not in body
    assert b'name="keywords"' not in body
    assert b'name="with_timestamps"\r\n\r\nfalse' in body
    assert result.segments is None
    assert result.metadata["audio_duration_secs"] == 1.2
    assert result.metadata["language_code"] == "ja"


@responses.activate
def test_transcribe_one_shot_maps_words_to_segments(fake_wav):
    payload = dict(STT_RESPONSE)
    payload["words"] = [
        {"text": "こんにちは", "type": "word", "logprob": 0.0, "start": 0.0, "end": 0.8, "speaker_id": None},
        {"text": " ", "type": "spacing", "logprob": 0.0, "start": None, "end": None, "speaker_id": None},
        {"text": "世界", "type": "word", "logprob": 0.0, "start": 0.9, "end": 1.2, "speaker_id": None},
    ]
    responses.add(responses.POST, STT_URL, json=payload, status=200)
    result = _client().asr.transcribe(fake_wav, with_timestamps=True)
    assert [(s.text, s.start, s.end) for s in result.segments] == [
        ("こんにちは", 0.0, 0.8),
        ("世界", 0.9, 1.2),
    ]
    assert "words" not in result.metadata  # timings live in segments; nothing is dumped raw
    assert result.metadata == {"language_code": "ja", "language_probability": 1.0, "audio_duration_secs": 1.2}


@responses.activate
def test_transcribe_one_shot_invalid_parameters_is_a_protocol_error(fake_wav):
    responses.add(
        responses.POST,
        STT_URL,
        json={"error": {"type": "invalid_request_error", "code": "invalid_parameters",
                        "message": "無効なパラメータです", "param": None}},
        status=400,
    )
    with pytest.raises(ProtocolError) as exc:
        _client().asr.transcribe(fake_wav)
    assert exc.value.status_code == 400
    assert str(exc.value) == "無効なパラメータです"


@responses.activate
def test_transcribe_one_shot_auth_error(fake_wav):
    responses.add(responses.POST, STT_URL, json={"detail": "Unauthorized"}, status=401)
    with pytest.raises(AuthError):
        _client().asr.transcribe(fake_wav)


@responses.activate
def test_transcribe_one_shot_worker_unavailable(fake_wav):
    responses.add(
        responses.POST,
        STT_URL,
        json={"error": {"type": "server_error", "code": "worker_unavailable",
                        "message": "worker_unavailable", "param": None}},
        status=503,
    )
    with pytest.raises(kotoba.APIError) as exc:
        _client().asr.transcribe(fake_wav)
    assert exc.value.status_code == 503


@responses.activate
def test_transcribe_one_shot_timeout(fake_wav):
    responses.add(responses.POST, STT_URL, body=requests.exceptions.ReadTimeout("slow"))
    with pytest.raises(kotoba.TimeoutError):
        kotoba.KotobaClient(api_key="token-abc", url=BASE_URL, max_retries=0, timeout=0.05).asr.transcribe(fake_wav)


@responses.activate
def test_transcribe_one_shot_on_fal_host(fake_wav):
    responses.add(responses.POST, "https://fal.run/team/app/v1/speech-to-text", json=STT_RESPONSE, status=200)
    client = kotoba.KotobaClient(api_key="fal-secret", url="https://fal.run/team/app/v1", max_retries=0)
    client.asr.transcribe(fake_wav)
    req = responses.calls[0].request
    assert req.headers["Authorization"] == "Key fal-secret"
    assert "xi-api-key" not in req.headers


async def test_async_transcribe_one_shot(fake_wav):
    seen = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        payload = dict(STT_RESPONSE)
        payload["words"] = [{"text": "こんにちは", "start": 0.0, "end": 0.8}]
        return httpx.Response(200, json=payload)

    session = AsyncHttpSession(base_url=BASE_URL, api_key="token-abc", timeout=5.0, max_retries=0)
    session._client = httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(app), headers={"Authorization": "Bearer token-abc"}
    )
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    result = await client.transcribe(fake_wav, language="ja", keywords=["a", "b"], with_timestamps=True)
    assert seen["path"] == "/v1/speech-to-text"
    assert "xi-api-key" not in seen["headers"]
    assert seen["body"].count(b'name="keywords"') == 2
    assert [(s.text, s.start, s.end) for s in result.segments] == [("こんにちは", 0.0, 0.8)]
    await session.aclose()


def _async_session_with(app, max_retries: int) -> AsyncHttpSession:
    session = AsyncHttpSession(
        base_url=BASE_URL, api_key="token-abc", timeout=5.0, max_retries=max_retries, backoff_factor=0.0
    )
    session._client = httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(app), headers={"Authorization": "Bearer token-abc"}
    )
    return session


async def test_async_post_is_sent_exactly_once_even_with_retries(fake_wav):
    """A timed-out or 5xx POST may already have been accepted (and billed), so the
    async session must not re-send it, unlike an idempotent GET."""
    calls = []

    def app(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(503, json={"error": {"code": "worker_unavailable", "message": "worker_unavailable"}})

    session = _async_session_with(app, max_retries=3)
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    with pytest.raises(kotoba.APIError) as exc:
        await client.transcribe(fake_wav)
    assert exc.value.status_code == 503
    assert calls == ["POST"]
    await session.aclose()


async def test_async_get_is_retried_on_transient_status():
    calls = []

    def app(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"state": "done", "transcription": "ok"})

    session = _async_session_with(app, max_retries=3)
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    status = await client.get_job("job-1")
    assert status.state == "done"
    assert calls == ["GET", "GET"]
    await session.aclose()


@responses.activate
def test_transcribe_rejects_batch_only_arguments_in_one_shot_mode(fake_wav):
    # No mock is registered: a request would fail loudly, so the ValueError proves nothing was sent.
    with pytest.raises(ValueError, match="poll_interval is only valid with batch=True"):
        _client().asr.transcribe(fake_wav, poll_interval=0.1)
    with pytest.raises(ValueError, match="timeout is only valid with batch=True"):
        _client().asr.transcribe(fake_wav, timeout=5.0)


@responses.activate
def test_transcribe_rejects_one_shot_arguments_in_batch_mode(fake_wav):
    with pytest.raises(ValueError, match="keywords is only valid with batch=False"):
        _client().asr.transcribe(fake_wav, batch=True, keywords=["Kotoba"])
    with pytest.raises(ValueError, match="file_format is only valid with batch=False"):
        _client().asr.transcribe(fake_wav, batch=True, file_format="pcm_s16le_16")


async def test_async_transcribe_rejects_cross_mode_arguments(fake_wav):
    session = _async_session_with(lambda request: httpx.Response(500), max_retries=0)
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    with pytest.raises(ValueError, match="keywords is only valid with batch=False"):
        await client.transcribe(fake_wav, batch=True, keywords=["Kotoba"])
    with pytest.raises(ValueError, match="poll_backoff is only valid with batch=True"):
        await client.transcribe(fake_wav, poll_backoff=2.0)
    await session.aclose()


@responses.activate
def test_transcribe_one_shot_does_not_follow_redirects(fake_wav):
    """A redirect must not be followed: requests would replay Authorization to another host."""
    responses.add(responses.POST, STT_URL, status=302, headers={"Location": "https://evil.example/speech-to-text"})
    responses.add(responses.POST, "https://evil.example/speech-to-text", json=STT_RESPONSE, status=200)
    with pytest.raises(kotoba.APIError) as exc:
        _client().asr.transcribe(fake_wav)
    assert exc.value.status_code == 302
    assert [call.request.url for call in responses.calls] == [STT_URL]  # the redirect target never saw the key


async def test_async_transcribe_one_shot_does_not_follow_redirects(fake_wav):
    seen = []

    def app(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://evil.example/speech-to-text"})

    session = _async_session_with(app, max_retries=3)
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    with pytest.raises(kotoba.APIError) as exc:
        await client.transcribe(fake_wav)
    assert exc.value.status_code == 302
    assert seen == [f"{BASE_URL}/speech-to-text"]
    await session.aclose()


@responses.activate
def test_transcribe_batch_without_timings_has_no_segments(fake_wav):
    """A done job with segments=[] (timestamp recovery off) maps to None, like the one-shot mode."""
    responses.add(responses.POST, JOBS_URL, json={"job_id": "job-empty"}, status=202)
    responses.add(
        responses.GET, f"{JOBS_URL}/job-empty", json={"state": "done", "transcription": "こんにちは", "segments": []}, status=200
    )
    result = _client().asr.transcribe(fake_wav, batch=True, with_timestamps=True, poll_interval=0.01)
    assert result.text == "こんにちは"
    assert result.segments is None


@responses.activate
def test_transcribe_one_shot_rejects_a_non_json_200(fake_wav):
    responses.add(responses.POST, STT_URL, body="<html>gateway</html>", status=200, content_type="text/html")

    with pytest.raises(ProtocolError, match="Non-JSON response body") as info:
        _client().asr.transcribe(fake_wav)

    assert info.value.status_code == 200
    assert info.value.payload == {"detail": "<html>gateway</html>"}


async def test_async_transcribe_one_shot_rejects_a_non_json_200(fake_wav):
    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    session = _async_session_with(app, max_retries=0)
    client = kotoba.AsyncASRClient(session, api_key="token-abc")
    with pytest.raises(ProtocolError, match="Non-JSON response body") as info:
        await client.transcribe(fake_wav)
    assert info.value.status_code == 200
    await session.aclose()
