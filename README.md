# kotoba-sdk

Python SDK for Kotoba Speech APIs:

- **REST** — async ASR transcription jobs (POST audio file, poll, get text)
- **WebSocket** — live streaming ASR, TTS, and S2ST (speech-to-speech translation)

## Install

```bash
pip install .
# or, from an editable checkout:
pip install -e .
```

Python ≥ 3.10. Dependencies: `requests`, `httpx`, `pydantic`, `websockets`, `numpy`, `soundfile`.

## Environment variables

The SDK reads configuration from these env vars only. There are no legacy
aliases — set exactly these names.

| Variable | Purpose |
|---|---|
| `KOTOBA_API_KEY` | Bearer token sent as `Authorization: Bearer …` (REST + WS) |
| `KOTOBA_ASR_REST_URL` | REST API base URL including version prefix, e.g. `https://xxx/v1` |
| `KOTOBA_ASR_URL` | WebSocket URL for live ASR streaming, e.g. `wss://yyy/v1/realtime` |
| `KOTOBA_TTS_JA_URL` | WebSocket URL for Japanese TTS, e.g. `wss://zzz/v2/tts/ws` |
| `KOTOBA_S2ST_EN_JA_URL` | WebSocket URL for English-to-Japanese speech translation |

## Quickstart

```python
import os
import kotoba

client = kotoba.KotobaClient(
    api_key=os.environ["KOTOBA_API_KEY"],
    url=os.environ["KOTOBA_ASR_REST_URL"],
)
result = client.asr.transcribe("sample.wav", language="ja")
print(result.text)
```

If `KOTOBA_API_KEY` and `KOTOBA_ASR_REST_URL` are set in the environment you
can drop both kwargs:

```python
client = kotoba.KotobaClient()
print(client.asr.transcribe("sample.wav").text)
```

### With timestamps

```python
result = client.asr.transcribe("sample.wav", with_timestamps=True)
print(result.text)
for seg in result.segments:
    print(f"{seg.start:6.2f} - {seg.end:6.2f}  {seg.text}")
```

### Async

```python
import asyncio, os, kotoba

async def main():
    async with kotoba.AsyncKotobaClient(
        api_key=os.environ["KOTOBA_API_KEY"],
        url=os.environ["KOTOBA_ASR_REST_URL"],
    ) as client:
        result = await client.asr.transcribe("sample.wav", language="ja")
        print(result.text)

asyncio.run(main())
```

### Live streaming ASR (WebSocket)

```python
import kotoba

client = kotoba.KotobaClient()  # reads KOTOBA_API_KEY + KOTOBA_ASR_URL
for delta in client.asr.transcribe_stream(file_chunk_iter("mic.pcm")):
    print(delta, end="", flush=True)
```

### TTS / S2ST

```python
client = kotoba.KotobaClient()  # reads KOTOBA_TTS_JA_URL
audio = client.tts.synthesize("こんにちは", language="ja", speaker_id="ja-man-1")
audio.to_wav("out.wav")

result = client.s2st.translate("input.wav", src="en", tgt="ja")  # reads KOTOBA_S2ST_EN_JA_URL
result.to_wav("translated.wav")
```

## Public API

### `kotoba.KotobaClient` / `kotoba.AsyncKotobaClient`

```python
KotobaClient(
    *,
    api_key: str | None = None,   # falls back to KOTOBA_API_KEY
    url: str | None = None,       # falls back to KOTOBA_ASR_REST_URL (REST only)
    timeout: float = 30.0,        # per-request HTTP timeout (s)
    max_retries: int = 3,         # for 429/5xx and network errors
)
```

Exposes:

- `.asr` — `ASRClient` / `AsyncASRClient` (REST + WS)
- `.tts` — `TTSClient` / `AsyncTTSClient` (WS)
- `.s2st` — `S2STClient` / `AsyncS2STClient` (WS)

The async variant supports `async with …` and exposes `await client.close()`.

### `client.asr.transcribe(...)` — high-level REST helper

```python
transcribe(
    audio_file_path: str | Path,
    *,
    language: str = "ja",
    with_timestamps: bool = False, # ask server for per-segment timestamps
    poll_interval: float = 1.0,    # initial GET polling interval (s)
    poll_backoff: float = 1.5,     # multiplied each poll
    max_poll_interval: float = 10.0,
    timeout: float = 1200.0,       # overall deadline for job completion
) -> TranscriptResult
```

POSTs the file, polls `GET /transcription_jobs/{id}` with exponential backoff,
returns the final transcript. Raises `TranscriptionError` on server-reported
failure, `TimeoutError` if the deadline elapses.

When `with_timestamps=True`, `TranscriptResult.segments` is populated with
`[Segment(text, start, end), ...]` (one per word/phrase chunk, derived from
the model's `<|pad|>` token grid and refined with silero-VAD on the server
side). Default is text-only; the server skips tokenizer + VAD work entirely.

### Low-level REST helpers

```python
client.asr.submit_job(path, language="ja") -> JobIDResponse  # POST
client.asr.get_job(job_id)                -> JobStatus       # GET, 202→processing
```

`JobStatus.state` is one of `JobState.processing | done | error`. For `done`,
read `.transcription`; for `error`, read `.error_message`.

### WebSocket entry points

```python
client.asr.stream(language="ja", url=...)           -> ASRSession
client.asr.transcribe_stream(audio_iter, ...)       -> Iterator[str]
client.asr.transcribe_file_ws(path, ...)            -> TranscriptResult

client.tts.stream(language="ja", speaker_id=..., url=...)  -> TTSSession
client.tts.synthesize_stream(text_or_iter, ...)            -> Iterator[bytes]
client.tts.synthesize(text, ...)                           -> AudioResult

client.s2st.stream(src="en", tgt="ja", url=...)  -> S2STSession
client.s2st.translate(path, src="en", tgt="ja")  -> S2STResult
```

URLs resolve from the per-route env vars (`KOTOBA_ASR_URL`,
`KOTOBA_TTS_JA_URL`, `KOTOBA_S2ST_EN_JA_URL`) unless passed explicitly with
`url=`. You can also register routes at runtime:

```python
from kotoba import register_endpoint
register_endpoint("tts", None, "ko", "wss://.../tts")
```

### Exceptions

All inherit from `kotoba.KotobaError`:

| Exception | When |
|---|---|
| `AuthError` | HTTP 401/403, WS auth rejection |
| `ProtocolError` | Other 4xx, or a server `error` frame violating the contract |
| `APIError` | Transport or 5xx that exhausted retries |
| `TimeoutError` | HTTP timeout, WS handshake timeout, or `transcribe()` polling deadline exceeded |
| `JobNotFoundError` | GET returned 404 |
| `TranscriptionError` | Job completed in `error` state |
| `UnsupportedRouteError` | No WS URL registered for the requested `(modality, src, tgt)` |

### Retry behavior (REST)

Both sync and async clients retry on network errors, 429, and 5xx with
exponential backoff. `Retry-After` headers on 429 are honored (async client).
4xx other than 429 raise immediately.

## Layout

```
src/kotoba/
  __init__.py    public exports
  client.py      KotobaClient / AsyncKotobaClient
  asr.py         REST + WS ASR client
  tts.py         WS TTS client
  s2st.py        WS speech-to-speech translation client
  _http.py       HttpSession / AsyncHttpSession (retry/backoff)
  _ws_*.py       per-modality WebSocket protocol handlers
  audio.py       PCM16 / PCM_F32 helpers
  routing.py     per-route env-var registry
  errors.py      typed exceptions
  models.py      pydantic models (TranscriptResult, StreamEvent, …)
```

Self-contained — can be copied to a standalone repo and built with `uv build`
or `pip install .`.
