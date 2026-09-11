# kotoba-sdk

Python SDK for Kotoba speech APIs — REST batch transcription and streaming **ASR**, **TTS**, and **speech-to-speech translation** over WebSockets.

> Phase-1 alpha. See [`docs/quickstart.md`](docs/quickstart.md).

## Install

```bash
pip install kotoba-sdk
```

Or from a checkout:

```bash
git clone https://github.com/kotoba-tech/kotoba-python.git
cd kotoba-python
uv venv
uv pip install -e .
```

Python ≥ 3.10. Optional `mic` extra (`pip install 'kotoba-sdk[mic]'`) installs `sounddevice` for live-microphone examples.

## Configure endpoints

The SDK reads configuration from these env vars only — set the ones for the routes you actually need:

| Variable | Purpose |
|---|---|
| `KOTOBA_API_KEY` | API key sent as `Authorization: Bearer …` (REST + WS); for `fal.run` URLs the SDK sends the same value as `Authorization: Key …`, so a fal key works unchanged |
| `KOTOBA_ASR_REST_URL` | REST API base URL including version prefix, e.g. `https://.../v1` |
| `KOTOBA_ASR_URL` | WebSocket URL for live ASR, e.g. `wss://.../asr` |
| `KOTOBA_TTS_URL` | WebSocket URL for TTS, e.g. `wss://.../v2/tts/ws`; the language is chosen per session |
| `KOTOBA_S2ST_URL` | WebSocket URL for speech-to-speech translation; the languages are chosen per session |

One URL per service is enough: the language (TTS) or language pair (S2ST) travels in the session, and a deployment serves the languages it was configured with. If one language is served by a separate deployment, register a language-specific route from code; it takes precedence over the service default for that language only:

```python
import kotoba
kotoba.register_endpoint("tts", None, "ko", "wss://.../ko-tts")  # only Korean goes here
```

URLs passed explicitly via `url=...` on a call take precedence over the registry.

### Using the Kotoba apps on fal

The same SDK talks to the Kotoba apps published on [fal](https://fal.ai). Use your fal key as
`KOTOBA_API_KEY` and point each service at its fal app; the `Key` auth scheme is selected
automatically for `fal.run` hosts.

```bash
export KOTOBA_API_KEY=<fal key>
export KOTOBA_ASR_REST_URL=https://fal.run/Kotoba-Technologies/kotoba-stt/v1
export KOTOBA_ASR_URL=wss://fal.run/Kotoba-Technologies/kotoba-streaming-stt/v1/realtime
export KOTOBA_TTS_URL=wss://fal.run/Kotoba-Technologies/kotoba-tts/v2/tts/ws
export KOTOBA_S2ST_URL=wss://fal.run/Kotoba-Technologies/kotoba-sts/v1/realtime_voice
```

A fal app that has been idle boots a GPU runner on the first request, which can take a few
minutes while the gateway holds the connection open. The SDK's defaults (30 s for REST via
`KotobaClient(timeout=...)`, 15 s for the WebSocket handshake) are shorter than that, so warm
the app first with `GET https://fal.run/Kotoba-Technologies/<app>/model_and_cuda_availability`
(any HTTP client, same `Authorization: Key` header) or retry the first call. The job API
(`transcribe(batch=True)`) is not available on fal; see the ASR table below.

## Quickstart

```python
import kotoba

client = kotoba.KotobaClient()  # reads KOTOBA_API_KEY + KOTOBA_*_URL from env

# 1) Speech recognition (REST, one request; batch=True for the job API)
result = client.asr.transcribe(
    "examples/audio/ja/example.mp3", language="ja"
)
print(result.text)

# 2) Text-to-Speech (Japanese, default speaker)
audio = client.tts.synthesize("こんにちは、世界。", language="ja")
audio.to_wav("hello.wav")

# 3) Speech-to-Speech translation (English -> Japanese)
translated = client.s2st.translate(
    "examples/audio/en/example.mp3", src="en", tgt="ja"
)
translated.to_wav("translated.wav")
print(translated.transcript_source)
```

`KotobaClient()` reads its credentials and URLs from env vars. Pass them explicitly if you'd rather not rely on the environment:

```python
client = kotoba.KotobaClient(
    api_key="sk_...",
    url="https://.../v1",                  # REST base
    asr_ws_url="wss://.../asr",
    tts_ws_url="wss://.../tts",
    s2st_ws_url="wss://.../sts",
)
```

## Streaming (the live surface)

ASR, TTS, and S2ST are all streaming-first. Audio chunks and partial transcripts surface the moment the server emits them, so you can play / display incrementally instead of waiting for the full response.

### Streaming output

ASR streams transcript deltas as audio arrives; TTS streams audio chunks
as the server produces them from a single text prompt. ASR accepts a
generator of PCM16 chunks on the input side (feed + drain run
concurrently); TTS sends the full text in one frame and streams audio
back:

```python
# ASR: pcm16 bytes in -> transcript deltas out
for delta in client.asr.transcribe_stream(mic_chunks(), language="ja"):
    print(delta, end="", flush=True)

# TTS: full text in -> pcm audio chunks streamed out
for pcm in client.tts.synthesize_stream("こんにちは、世界。", language="ja"):
    speaker.write(pcm)
```

### Async (recommended for production)

```python
import asyncio, kotoba

async def main():
    client = kotoba.AsyncKotobaClient()

    async with client.tts.stream(language="ja") as session:
        await session.synthesize("こんにちは。本日はよろしくお願いします。")

        async for event in session:
            if event.type == "audio_chunk":
                await play(event.audio)
            elif event.type == "done":
                break

asyncio.run(main())
```

### Sync (notebooks, scripts)

```python
import kotoba

client = kotoba.KotobaClient()
with client.s2st.stream(src="en", tgt="ja") as session:
    for chunk in pcm16_chunks_from_mic():
        session.send_audio(chunk)
    session.commit()
    for event in session:
        if event.type == "partial_transcript":
            print(event.text, end="", flush=True)
        elif event.type == "audio_chunk":
            speaker.write(event.audio)
        elif event.type == "done":
            break
```

The sync wrapper runs an `asyncio` loop on a background daemon thread, so the underlying transport is identical — only the call style differs.

## What's in the box

| Module | What |
|---|---|
| `kotoba.KotobaClient` / `AsyncKotobaClient` | Top-level entry point |
| `client.asr.transcribe(path, ...)` | **REST** transcription — one `POST /v1/speech-to-text` by default (the `stt` app; on fal `kotoba-stt`); `batch=True` submits a job and polls (self-hosted `streaming_stt` only); optional `with_timestamps=True` |
| `client.asr.stream(...)` / `transcribe_stream(iter)` | Streaming ASR (Japanese, English) over WebSocket |
| `client.tts.stream(...)` / `synthesize(...)` / `synthesize_stream(...)` | Streaming TTS (Japanese) |
| `client.s2st.stream(...)` / `translate(...)` | Streaming speech-to-speech translation |
| `kotoba.register_endpoint(...)` | Route one language (pair) of a service to a different deployment |
| `kotoba.audio.*` | PCM16 / float32 WAV helpers |

## Examples

Each example under `examples/` is runnable with `uv run examples/<file>.py` and uses bundled audio under `examples/audio/` by default.

Naming: `_sync` / `_async` is the client class (`KotobaClient` vs `AsyncKotobaClient`), not the server call; `_batch` is the job API (`transcribe(..., batch=True)`). `asr_transcribe_async.py` sends the same single `POST /v1/speech-to-text` as the sync file, just from asyncio.

| File | What it shows | Required env |
|---|---|---|
| `asr_transcribe_sync.py` | `transcribe()` against an `stt` deployment (one request), optional `--timestamps` | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_transcribe_async.py` | Same, async with the `AsyncKotobaClient` context manager | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_transcribe_batch_sync.py` | Job API, `transcribe(..., batch=True)` with `with_timestamps=True` (self-hosted `streaming_stt` only) | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_transcribe_batch_async.py` | Same, async with the `AsyncKotobaClient` context manager | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_stream_async.py` | Live ASR via `transcribe_stream(generator)` with first-token-latency measurement | `KOTOBA_API_KEY`, `KOTOBA_ASR_URL` |
| `tts_synthesize_sync.py` | One-shot TTS with explicit `speaker_id` | `KOTOBA_API_KEY`, `KOTOBA_TTS_URL` |
| `tts_stream_async.py` | One-shot text in → streamed audio chunks with first-audio-latency timing | `KOTOBA_API_KEY`, `KOTOBA_TTS_URL` |
| `s2st_stream_async.py` | File in → live transcript + translated WAV out | `KOTOBA_API_KEY`, `KOTOBA_S2ST_URL` |
| `s2st_mic_async.py` | **Live microphone** in → translated WAV out (Ctrl-C to stop). Requires `pip install 'kotoba-sdk[mic]'` and PortAudio. | `KOTOBA_API_KEY`, `KOTOBA_S2ST_URL` |

REST is shown in both sync + async because the context-manager pattern matters for resource cleanup. Streaming examples are async-by-default — wrap with `kotoba.KotobaClient()` for sync (the snippets above show the conversion).

## Public API

### `kotoba.KotobaClient` / `kotoba.AsyncKotobaClient`

```python
KotobaClient(
    *,
    api_key: str | None = None,           # KOTOBA_API_KEY
    url: str | None = None,               # KOTOBA_ASR_REST_URL  (REST)
    asr_ws_url: str | None = None,        # KOTOBA_ASR_URL       (WS ASR)
    tts_ws_url: str | None = None,        # KOTOBA_TTS_URL       (WS TTS; language per session)
    s2st_ws_url: str | None = None,       # KOTOBA_S2ST_URL      (WS S2ST; languages per session)
    timeout: float = 30.0,                # per-request HTTP timeout (s)
    max_retries: int = 3,                 # GET only (429/5xx, network errors); a POST is sent once
)
```

Exposes:

- `.asr` — `ASRClient` / `AsyncASRClient` (REST + WS)
- `.tts` — `TTSClient` / `AsyncTTSClient` (WS)
- `.s2st` — `S2STClient` / `AsyncS2STClient` (WS)

The async variant supports `async with …` and exposes `await client.close()`.

### `client.asr.transcribe(...)` — REST transcription

```python
transcribe(
    audio_file_path: str | Path,
    *,
    batch: bool = False,                # False: one POST /v1/speech-to-text; True: submit + poll /v1/transcription_jobs
    language: str | None = None,        # batch=False: None = the server's configured language; batch=True: None = "ja"
    with_timestamps: bool = False,      # per-segment timestamps (batch=False: must be enabled on the deployment)
    style_preference: TranscriptionStylePreference | dict | None = None,  # {"human_name": "kana"}
    keywords: list[str] | None = None,  # legacy hotword biasing; not supported on current deployments (rejected with 400)
    file_format: "other" | "pcm_s16le_16" | None = None,  # batch=False only; raw 16 kHz mono int16, or any encoded audio
    poll_interval: float | None = None,     # batch=True only (default 1.0 s)
    poll_backoff: float | None = None,      # batch=True only (default 1.5, multiplied each poll)
    max_poll_interval: float | None = None, # batch=True only (default 10.0 s)
    timeout: float | None = None,           # batch=True only: overall deadline for job completion (default 1200 s)
) -> TranscriptResult
```

`batch=False` (default) uploads the file in a single synchronous `POST /v1/speech-to-text` and returns the transcript — no job, no polling — for `stt` deployments (on fal `kotoba-stt`). The server's `max_audio_seconds` limit (120 s by default) applies; longer audio, and options the deployment has not enabled, come back as a `ProtocolError` (HTTP 400). Words with timings map to `TranscriptResult.segments`; `language_code` / `language_probability` / `audio_duration_secs` are in `TranscriptResult.metadata`.

`batch=True` POSTs the file to `/transcription_jobs`, polls `GET /transcription_jobs/{id}` with exponential backoff and returns the final transcript (`job_id` set; `segments` when `with_timestamps=True` and the deployment produced timings, else `None`). Raises `TranscriptionError` on server-reported failure, `TimeoutError` if the deadline elapses. Self-hosted `streaming_stt` deployments only.

Passing an argument that belongs to the other mode raises `ValueError` before any request is sent.

#### Which URL to set for which call

<!-- The server-side contract (which app mounts which endpoint) is specified in
     .agent/sdd/kotoba-api-spec.md; this table is the SDK view of it. Keep the two in sync. -->

| SDK call | Set | to a deployment of | On fal |
|---|---|---|---|
| `transcribe()` | `KOTOBA_ASR_REST_URL` | `stt` | `kotoba-stt` |
| `transcribe(batch=True)` | `KOTOBA_ASR_REST_URL` | `streaming_stt` (self-hosted only) | not supported |
| `stream()` / `transcribe_stream()` | `KOTOBA_ASR_URL` | `streaming_stt` | `kotoba-streaming-stt` |

**Batch ASR is not supported on fal.** `transcribe(batch=True)` (the `/v1/transcription_jobs` job API) exists only on self-hosted `streaming_stt` deployments; on fal use `transcribe()` for files and `stream()` / `transcribe_stream()` for live audio. The deployed app decides which endpoints exist; the wrong app answers 401 or 404. The default `transcribe()` runs under `KotobaClient(timeout=...)` (30 s by default); long files or a cold deployment may need more.

### Low-level REST helpers

```python
client.asr.submit_job(path, language="ja") -> JobIDResponse  # POST
client.asr.get_job(job_id)                -> JobStatus       # GET, 202→processing
```

`JobStatus.state` is one of `JobState.processing | done | error`. For `done`, read `.transcription`; for `error`, read `.error_message`.

### WebSocket entry points

```python
client.asr.stream(language="ja", sample_rate=24000, keywords=None, style_preference=None, url=...)  -> ASRSession
client.asr.transcribe_stream(audio_iter, language="ja", sample_rate=24000, keywords=None, style_preference=None, url=...)  -> Iterator[str]

client.tts.stream(language="ja", speaker_id=..., url=...)  -> TTSSession
client.tts.synthesize_stream(text, ...)                    -> Iterator[bytes]
client.tts.synthesize(text, ...)                           -> AudioResult

client.s2st.stream(src="en", tgt="ja", url=...)  -> S2STSession
client.s2st.translate(path, src="en", tgt="ja")  -> S2STResult
```

URLs resolve from the per-service env vars (`KOTOBA_ASR_URL`, `KOTOBA_TTS_URL`, `KOTOBA_S2ST_URL`) or the matching `KotobaClient(...)` kwargs, then from any language-specific `register_endpoint(...)` route, unless passed explicitly with `url=`.

Streaming STT takes the same recognition options as `transcribe()`: `style_preference={"human_name": "kana"}` (personal names in katakana) and the legacy `keywords` (not supported on current deployments). Both are sent once in the session-opening `transcription_session.update` message and apply to the whole session. Note that `/v1/realtime` does not reject options a deployment has not enabled; keyword biasing that is off on the worker is silently ignored there, whereas `POST /v1/speech-to-text` answers 400.

### Exceptions

All inherit from `kotoba.KotobaError`:

| Exception | When |
|---|---|
| `AuthError` | HTTP 401/403, WS auth rejection |
| `ProtocolError` | Other 4xx, a server `error` frame violating the contract, or a 200 whose body does not match the `/speech-to-text` response contract (`status_code=200`) |
| `APIError` | Transport error, 5xx (after GET retries), or an unexpected redirect |
| `TimeoutError` | HTTP timeout, WS handshake timeout, or `transcribe(batch=True)` polling deadline exceeded |
| `JobNotFoundError` | GET returned 404 |
| `TranscriptionError` | Job completed in `error` state |
| `UnsupportedRouteError` | No WS URL registered for the requested `(modality, src, tgt)` |

### Retry behavior (REST)

Both sync and async clients retry **GET** requests on network errors, 429 and 5xx with exponential backoff (`Retry-After` on 429 is honored by the async client); 4xx other than 429 raise immediately. A **POST** is sent exactly once: an upload that timed out may already have been accepted, transcribed and billed, so re-sending it could duplicate the work and the charge. Redirects are never followed; a 3xx raises `APIError`, so the `Authorization` credential is never replayed to another host.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```
