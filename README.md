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
| `KOTOBA_API_KEY` | Bearer token sent as `Authorization: Bearer …` (REST + WS) |
| `KOTOBA_ASR_REST_URL` | REST API base URL including version prefix, e.g. `https://.../v1` |
| `KOTOBA_ASR_URL` | WebSocket URL for live ASR, e.g. `wss://.../asr` |
| `KOTOBA_TTS_JA_URL` | WebSocket URL for Japanese TTS, e.g. `wss://.../tts` |
| `KOTOBA_S2ST_EN_JA_URL` | WebSocket URL for English-to-Japanese speech translation |

You can also register routes from code:

```python
import kotoba
kotoba.register_endpoint("tts", None, "ko", "wss://.../tts")
```

URLs passed explicitly via `url=...` on a call take precedence over the registry.

## Quickstart

```python
import kotoba

client = kotoba.KotobaClient()  # reads KOTOBA_API_KEY + KOTOBA_*_URL from env

# 1) Speech recognition (REST batch — default for files)
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
    tts_ja_ws_url="wss://.../tts",
    s2st_en_ja_ws_url="wss://.../sts",
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
| `client.asr.transcribe(path, ...)` | **REST** batch transcription with optional `with_timestamps=True` |
| `client.asr.stream(...)` / `transcribe_stream(iter)` | Streaming ASR (Japanese, English) over WebSocket |
| `client.tts.stream(...)` / `synthesize(...)` / `synthesize_stream(...)` | Streaming TTS (Japanese) |
| `client.s2st.stream(...)` / `translate(...)` | Streaming speech-to-speech translation |
| `kotoba.register_endpoint(...)` | Add `(modality, src, tgt) -> URL` routes |
| `kotoba.audio.*` | PCM16 / float32 WAV helpers |

## Examples

Each example under `examples/` is runnable with `uv run examples/<file>.py` and uses bundled audio under `examples/audio/` by default.

| File | What it shows | Required env |
|---|---|---|
| `asr_rest_sync.py` | REST batch transcription with `with_timestamps=True`, sync | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_rest_async.py` | Same, async with `AsyncKotobaClient` context manager | `KOTOBA_API_KEY`, `KOTOBA_ASR_REST_URL` |
| `asr_stream_async.py` | Live ASR via `transcribe_stream(generator)` with first-token-latency measurement | `KOTOBA_API_KEY`, `KOTOBA_ASR_URL` |
| `tts_synthesize_sync.py` | One-shot TTS with explicit `speaker_id` | `KOTOBA_API_KEY`, `KOTOBA_TTS_JA_URL` |
| `tts_stream_async.py` | One-shot text in → streamed audio chunks with first-audio-latency timing | `KOTOBA_API_KEY`, `KOTOBA_TTS_JA_URL` |
| `s2st_stream_async.py` | File in → live transcript + translated WAV out | `KOTOBA_API_KEY`, `KOTOBA_S2ST_EN_JA_URL` |
| `s2st_mic_async.py` | **Live microphone** in → translated WAV out (Ctrl-C to stop). Requires `pip install 'kotoba-sdk[mic]'` and PortAudio. | `KOTOBA_API_KEY`, `KOTOBA_S2ST_EN_JA_URL` |

REST is shown in both sync + async because the context-manager pattern matters for resource cleanup. Streaming examples are async-by-default — wrap with `kotoba.KotobaClient()` for sync (the snippets above show the conversion).

## Public API

### `kotoba.KotobaClient` / `kotoba.AsyncKotobaClient`

```python
KotobaClient(
    *,
    api_key: str | None = None,           # KOTOBA_API_KEY
    url: str | None = None,               # KOTOBA_ASR_REST_URL  (REST)
    asr_ws_url: str | None = None,        # KOTOBA_ASR_URL       (WS ASR)
    tts_ja_ws_url: str | None = None,     # KOTOBA_TTS_JA_URL    (WS TTS)
    s2st_en_ja_ws_url: str | None = None, # KOTOBA_S2ST_EN_JA_URL
    timeout: float = 30.0,                # per-request HTTP timeout (s)
    max_retries: int = 3,                 # for 429/5xx and network errors
)
```

Exposes:

- `.asr` — `ASRClient` / `AsyncASRClient` (REST + WS)
- `.tts` — `TTSClient` / `AsyncTTSClient` (WS)
- `.s2st` — `S2STClient` / `AsyncS2STClient` (WS)

The async variant supports `async with …` and exposes `await client.close()`.

### `client.asr.transcribe(...)` — REST batch helper

```python
transcribe(
    audio_file_path: str | Path,
    *,
    language: str = "ja",
    with_timestamps: bool = False,  # ask server for per-segment timestamps
    poll_interval: float = 1.0,     # initial GET polling interval (s)
    poll_backoff: float = 1.5,      # multiplied each poll
    max_poll_interval: float = 10.0,
    timeout: float = 1200.0,        # overall deadline for job completion
) -> TranscriptResult
```

POSTs the file, polls `GET /transcription_jobs/{id}` with exponential backoff, returns the final transcript. Raises `TranscriptionError` on server-reported failure, `TimeoutError` if the deadline elapses.

When `with_timestamps=True`, `TranscriptResult.segments` is populated with `[Segment(text, start, end), …]`.

### Low-level REST helpers

```python
client.asr.submit_job(path, language="ja") -> JobIDResponse  # POST
client.asr.get_job(job_id)                -> JobStatus       # GET, 202→processing
```

`JobStatus.state` is one of `JobState.processing | done | error`. For `done`, read `.transcription`; for `error`, read `.error_message`.

### WebSocket entry points

```python
client.asr.stream(language="ja", url=...)           -> ASRSession
client.asr.transcribe_stream(audio_iter, ...)       -> Iterator[str]

client.tts.stream(language="ja", speaker_id=..., url=...)  -> TTSSession
client.tts.synthesize_stream(text, ...)                    -> Iterator[bytes]
client.tts.synthesize(text, ...)                           -> AudioResult

client.s2st.stream(src="en", tgt="ja", url=...)  -> S2STSession
client.s2st.translate(path, src="en", tgt="ja")  -> S2STResult
```

URLs resolve from the per-route env vars (`KOTOBA_ASR_URL`, `KOTOBA_TTS_JA_URL`, `KOTOBA_S2ST_EN_JA_URL`) unless passed explicitly with `url=`.

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

Both sync and async clients retry on network errors, 429, and 5xx with exponential backoff. `Retry-After` headers on 429 are honored (async client). 4xx other than 429 raise immediately.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```
