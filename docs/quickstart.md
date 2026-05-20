# Kotoba SDK Quickstart

Goal: from zero to a working transcript / synthesized audio in ~10 minutes.

## 1. Get an API key

Request a sandbox key from your Kotoba contact. Set it — along with the endpoints for the modalities you plan to use — in your shell:

```bash
export KOTOBA_API_KEY=sk_...
export KOTOBA_ASR_REST_URL=https://.../v1            # batch ASR (default transport)
export KOTOBA_ASR_URL=wss://.../asr                  # live streaming ASR
export KOTOBA_S2ST_EN_JA_URL=wss://.../sts           # speech-to-speech en -> ja
export KOTOBA_TTS_JA_URL=wss://.../tts               # streaming TTS (ja)
```

Only the routes you actually call need to be set. You can also register routes from code with `kotoba.register_endpoint(modality, src, tgt, url)`, or pass `url=...` directly to any `stream(...)` / `transcribe(...)` / `synthesize(...)` call.

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

- **REST** (`client.asr.transcribe`) — POST + poll. Default for batch / file-based work. Best for long files, scales naturally on the server, supports per-segment timestamps.
- **WebSocket** (`client.asr.stream` / `client.asr.transcribe_stream`) — push PCM16 chunks, read partial transcripts as they arrive. Best for live mic / latency-sensitive pipelines.

### REST batch

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

`transcribe()` accepts any audio format `soundfile` can decode (WAV / FLAC / OGG / MP3 / …) — the SDK uploads the file as-is and the server does the heavy lifting.

### Streaming (live mic)

For the realtime / mic case — where you want transcript deltas to surface *while* audio is still being captured — pass a generator directly to `transcribe_stream(...)`. The feeder and receiver run concurrently, so the first delta can fire before your source is exhausted:

```python
for delta in client.asr.transcribe_stream(mic_chunks(), language="ja"):
    print(delta, end="", flush=True)
```

`mic_chunks()` is any iterable of PCM16 little-endian mono bytes — it is **not** provided by the SDK. For a runnable end-to-end example (file-driven generator that mimics a live mic, with first-token-latency measurement), see [`examples/asr_stream_async.py`](../examples/asr_stream_async.py).

Optional knobs on both `stream(...)` and `transcribe_stream(...)`:

- `language`: `"ja"` or `"en"`.
- `sample_rate`: defaults to 24 kHz; the session resamples internally if your capture rate differs.
- `keywords`: list of hotword biases, e.g. `["Kotobatech", "LLM"]`.

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

- `examples/` for runnable demos (REST + streaming + mic). Each example has a default audio path under `examples/audio/`, so `uv run examples/asr_rest_sync.py` works without arguments.
- `kotoba.register_endpoint(modality, src, tgt, url)` if your routes aren't in the default registry yet.
- API reference: imports under `kotoba.*`.

## Notes / current limitations

- The SDK has no built-in endpoint defaults: every route must come from a `KOTOBA_*_URL` env var, a `register_endpoint(...)` call, or an explicit `url=...` argument. For language pairs beyond the env-var set, register them:

  ```python
  kotoba.register_endpoint("tts", None, "ko", "wss://your-ko-tts-host/ws")
  ```

- WebSocket ASR accepts PCM16 LE mono audio. `client.asr.transcribe(path)` (REST) decodes any `soundfile`-readable format; for `stream(...)` the caller is responsible for providing raw PCM16 bytes.
- TTS audio is emitted as `pcm_f32` @ 24 kHz mono; `result.to_wav()` converts to a playable int16 WAV automatically.
