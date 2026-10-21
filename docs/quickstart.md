# Kotoba SDK Quickstart

Goal: from zero to a working transcript / synthesized audio in ~10 minutes.

## 1. Get an API key

Request a sandbox key from your Kotoba contact, or use a fal key with the Kotoba apps on fal (see the README section "Using the Kotoba apps on fal" for the fal URLs). Set it — along with the endpoints for the modalities you plan to use — in your shell:

```bash
export KOTOBA_API_KEY=sk_...
export KOTOBA_ASR_REST_URL=https://.../v1            # REST speech-to-text (transcribe())
export KOTOBA_ASR_URL=wss://.../asr                  # live streaming ASR
export KOTOBA_S2ST_URL=wss://.../sts                 # speech-to-speech translation (languages per session)
export KOTOBA_TTS_URL=wss://.../tts                  # streaming TTS (language per session)
```

Only the services you actually call need to be set; one URL per service covers every language, since the language is chosen per session. If one language is served by a separate deployment, route it with `kotoba.register_endpoint(modality, src, tgt, url)`, or pass `url=...` directly to any `stream(...)` / `transcribe(...)` / `synthesize(...)` call.

## 2. Install

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

## 3. Hello World — TTS in 5 lines

```python
import kotoba

client = kotoba.KotobaClient()
result = client.tts.synthesize("こんにちは、世界。", language="ja")
result.to_wav("hello.wav")
```

Open `hello.wav` in any audio player. Done.

Need a specific wire format? Pass `audio_format=` (and, for PCM, `sample_rate=` — one of 8000 / 16000 / 24000). For example, `client.tts.synthesize("...", language="ja", audio_format="mulaw")` returns 8 kHz G.711 mu-law, the format Twilio's Media Streams expect; `result.to_wav()` still writes a playable WAV. `audio_format="opus"` returns a self-contained 24 kHz Ogg/Opus stream — write `result.data` straight to a `.ogg` file (it is not WAV-convertible).

## 4. Streaming TTS (incremental playback)

The streaming API yields audio chunks as the server produces them, so you can play (or send to a speaker / WebRTC track) without waiting for the full response.

```python
import kotoba

client = kotoba.KotobaClient()
with client.tts.stream(language="ja") as session:
    session.synthesize("こんにちは。本日はよろしくお願いします。")

    for event in session:
        if event.type == "audio_chunk":
            # event.audio is float32 PCM @ 24 kHz; pipe to a speaker
            handle(event.audio)
        elif event.type == "done":
            break
```

Async version: replace `with` with `async with` and add `await` to each call. See `examples/tts_stream_async.py` for a runnable end-to-end demo with first-audio-latency timing.

## 5. Speech recognition (ASR)

ASR has two transports. Pick by use case:

- **REST** (`client.asr.transcribe`) — one `POST /v1/speech-to-text` per file to an `stt` deployment (fal: `kotoba-stt`). Audio over the server's `max_audio_seconds` (120 s by default) is transcribed in chunks, up to `max_request_audio_seconds` (24 h by default); per-segment timestamps on request. Self-hosted `streaming_stt` deployments also offer `transcribe(..., batch=True)`, an asynchronous job API.
- **WebSocket** (`client.asr.stream` / `client.asr.transcribe_stream`) — push PCM16 chunks, read partial transcripts as they arrive. Best for live mic / latency-sensitive pipelines.

### REST (file transcription)

```python
import kotoba

client = kotoba.KotobaClient()
result = client.asr.transcribe("clip.mp3", language="ja")
print(result.text)
```

With per-segment timestamps:

```python
result = client.asr.transcribe("clip.mp3", language="ja", with_timestamps=True)
print(result.text)
for seg in result.segments:
    print(f"{seg.start:6.2f} - {seg.end:6.2f}  {seg.text}")
```

With speaker diarization (deployments whose model carries speaker tokens, e.g. the diarized `kotoba-stt` on fal):

```python
result = client.asr.transcribe("meeting.wav", language="ja", diarization=True, with_timestamps=True)
for u in result.utterances:
    print(f"{u.start:6.2f}  {u.speaker_id}: {u.text}")
```

`utterances` groups consecutive words by speaker in reading order. Without `with_timestamps=True` the times are `None`; a run the model did not attribute has `speaker_id=None`.

`transcribe()` accepts any audio format `soundfile` can decode (WAV / FLAC / OGG / MP3 / …) — the SDK uploads the file as-is and the server does the heavy lifting.

`transcribe()` sends one `POST /v1/speech-to-text` to an `stt` deployment (fal: `kotoba-stt`), subject to the server's audio-length limit (`max_request_audio_seconds`, 24 h by default; longer than `max_audio_seconds`, 120 s, is transcribed in chunks). Self-hosted `streaming_stt` deployments offer the asynchronous job API instead: `client.asr.transcribe("clip.mp3", batch=True)` submits the file and polls until the job finishes. On fal, only the default mode and the WebSocket `/v1/realtime` are available; the job API is not.

### Streaming (live mic)

For the realtime / mic case — where you want transcript deltas to surface *while* audio is still being captured — pass a generator directly to `transcribe_stream(...)`. The feeder and receiver run concurrently, so the first delta can fire before your source is exhausted:

```python
for delta in client.asr.transcribe_stream(
    mic_chunks(), language="ja", turn_detection=kotoba.ServerVAD(silence_duration_ms=800)
):
    print(delta, end="", flush=True)
```

`mic_chunks()` is any iterable of PCM16 little-endian mono bytes — it is **not** provided by the SDK. For a runnable end-to-end example (file-driven generator that mimics a live mic, with first-token-latency measurement), see [`examples/asr_stream_async.py`](../examples/asr_stream_async.py).

Optional knobs on both `stream(...)` and `transcribe_stream(...)`:

- `language`: `"ja"` or `"en"`.
- `sample_rate`: defaults to 24 kHz; the session resamples internally if your capture rate differs.
- `keywords`: terms whose spelling the transcript should prefer (names, product words); the deployment sets the limits (1,000 terms of 64 characters on fal today).
- `style_preference`: `{"human_name": "kana"}` to transcribe personal and place names in katakana.

For `turn_detection` configuration, defaults, and completed-turn events, see the
[SDK contract](../../../../.agent/sdd/kotoba-api-spec.md#python-sdk-streaming-asr).

## 6. Speech-to-Speech translation

```python
import kotoba

client = kotoba.KotobaClient()
result = client.s2st.translate("clip.mp3", src="en", tgt="ja")
result.to_wav("translated.wav")
print("source transcript:", result.transcript_source)
```

For incremental transcripts and audio out (e.g. live captioning), use `client.s2st.stream(...)`. See `examples/s2st_stream_async.py` for a file-driven demo, or `examples/s2st_mic_async.py` for live-microphone input (requires `pip install 'kotoba-sdk[mic]'`).

## 7. Where to go next

- `examples/` for runnable demos (REST + streaming + mic). Each example has a default audio path under `examples/audio/`, so `uv run examples/asr_transcribe_sync.py` works without arguments.
- `kotoba.register_endpoint(modality, src, tgt, url)` to send one language (pair) of a service to a different deployment.
- API reference: imports under `kotoba.*`.

## Notes / current limitations

- The SDK has no built-in endpoint defaults: every service URL must come from a `KOTOBA_*_URL` env var, a `KotobaClient(...)` kwarg, a `register_endpoint(...)` call, or an explicit `url=...` argument. A language served by a separate deployment gets its own route:

  ```python
  kotoba.register_endpoint("tts", None, "ko", "wss://your-ko-tts-host/ws")  # only Korean goes here
  ```

- WebSocket ASR accepts PCM16 LE mono audio. `client.asr.transcribe(path)` (REST) decodes any `soundfile`-readable format; for `stream(...)` the caller is responsible for providing raw PCM16 bytes.
- TTS audio defaults to `pcm_f32` @ 24 kHz mono; `result.to_wav()` converts to a playable int16 WAV automatically. Negotiate other formats with `audio_format=` — `pcm16` / `float32` (`sample_rate` 8/16/24 kHz), `mulaw` (8 kHz G.711, WAV-convertible), or `opus` (24 kHz Ogg/Opus, save the raw bytes).
