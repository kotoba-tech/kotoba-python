"""Live smoke test for the four Fal-deployed Kotoba apps through the SDK.

Runs (in order): TTS -> streaming STT -> batch STT (REST) -> S2ST.
The TTS output is reused as the speech input for the STT tests (falls back
to a sine tone if TTS didn't run).

Usage:

    export FAL_KEY=...
    export FAL_TEAM=<your-fal-team>        # e.g. the team in fal.run/<team>/<app>
    export FAL_ENV=<deploy-env>            # optional --<env> app suffix
    python examples/fal_smoke.py           # all four
    python examples/fal_smoke.py tts stt   # a subset

Configuration (all via environment):

    FAL_TEAM               fal team name; used to build default app URLs
    FAL_ENV                optional environment suffix (app name --<env>)
    FAL_TTS_URL            full wss:// URL for the TTS app (overrides default)
    FAL_STREAMING_STT_URL  full wss:// URL for the streaming STT app
    FAL_STT_REST_URL       REST base URL (app root) for the batch STT app
    FAL_STT_PROBE_PATH     readiness path probed before REST submits
                           (default /model_and_cuda_availability)
    FAL_S2ST_URL           full wss:// URL for the S2ST app
    FAL_S2ST_INPUT         path to a speech file (wav/flac/ogg) in the SOURCE
                           language for the S2ST test. Without it the test
                           falls back to TTS output or a sine tone, and the
                           model correctly outputs silence for those — only
                           the plumbing is verified.
    FAL_S2ST_SRC           source language for S2ST (default en)
    FAL_S2ST_TGT           target language for S2ST (default ja)
    FAL_PROBE_DEADLINE     REST probe deadline in seconds (default 600)
    FAL_WS_DEADLINE        WS session-init retry deadline in seconds
                           (default 360) — used by tts/streaming-stt/s2st

If the REST test ends in WorkerStartupError while the app is clearly up,
the probe path is wrong for your deployment — adjust FAL_STT_REST_URL /
FAL_STT_PROBE_PATH (the probe URL is base + path).
"""

from __future__ import annotations

import argparse
import array
import logging
import math
import os
import struct
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path

import kotoba

# Show each cold-start retry / readiness probe with its underlying error.
logging.basicConfig(
    level=logging.INFO, format="  [%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
)
logging.getLogger("websockets").setLevel(logging.WARNING)

TTS_TEXT = "こんにちは。コトバの音声APIの接続テストです。"
OUT_DIR = Path("fal_smoke_out")
SPEECH_WAV = OUT_DIR / "tts_speech.wav"


def _url(env_name: str, app: str, scheme: str, path: str) -> str:
    explicit = os.environ.get(env_name)
    if explicit:
        return explicit
    team = os.environ.get("FAL_TEAM")
    if not team:
        raise SystemExit(
            f"Set {env_name} (full URL) or FAL_TEAM (+ optional FAL_ENV) "
            f"to locate the {app} app"
        )
    env = os.environ.get("FAL_ENV")
    suffix = f"--{env}" if env else ""
    return f"{scheme}://fal.run/{team}/{app}{suffix}{path}"


def _tts_url() -> str:
    return _url("FAL_TTS_URL", "kotoba-tts", "wss", "/v2/tts/ws")


def _streaming_stt_url() -> str:
    return _url("FAL_STREAMING_STT_URL", "kotoba-streaming-stt", "wss", "/v1/realtime")


def _stt_rest_url() -> str:
    return _url("FAL_STT_REST_URL", "kotoba-stt", "https", "")


def _s2st_url() -> str:
    return _url("FAL_S2ST_URL", "kotoba-sts", "wss", "/v1/realtime_voice")


def _provider() -> kotoba.ProviderConfig:
    """The FAL profile with deadlines tunable from the environment."""
    probe = replace(
        kotoba.FAL.http_probe,
        probe_path=os.environ.get("FAL_STT_PROBE_PATH", kotoba.FAL.http_probe.probe_path),
        deadline_s=float(os.environ.get("FAL_PROBE_DEADLINE", "600")),
    )
    ws_retry = replace(
        kotoba.FAL.ws_retry,
        deadline_s=float(os.environ.get("FAL_WS_DEADLINE", "360")),
    )
    return replace(kotoba.FAL, http_probe=probe, ws_retry=ws_retry)


def _client(**kwargs) -> kotoba.KotobaClient:
    return kotoba.KotobaClient(provider=_provider(), **kwargs)  # key from FAL_KEY


def _write_pcm16_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)


def _f32_to_pcm16(data: bytes) -> bytes:
    floats = array.array("f", data)
    ints = array.array(
        "h", (int(max(-1.0, min(1.0, x)) * 32767) for x in floats)
    )
    return ints.tobytes()


def _sine_wav(path: Path, seconds: float = 2.0, rate: int = 24000) -> None:
    n = int(seconds * rate)
    pcm = b"".join(
        struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n)
    )
    _write_pcm16_wav(path, pcm, rate)


def _speech_wav() -> Path:
    """The TTS test's output if it ran, else a sine tone (plumbing-only)."""
    if not SPEECH_WAV.exists():
        print("  (no TTS output available; using a 440 Hz sine tone as input)")
        _sine_wav(SPEECH_WAV)
    return SPEECH_WAV


def _wav_chunks(path: Path, chunk_ms: int = 200):
    with wave.open(str(path), "rb") as w:
        frames_per_chunk = int(w.getframerate() * chunk_ms / 1000)
        while True:
            data = w.readframes(frames_per_chunk)
            if not data:
                return
            yield data


# ---------- the four tests --------------------------------------------------


def test_tts() -> None:
    url = _tts_url()
    print(f"  url: {url}")
    session = _client().tts.stream(language="ja", url=url)
    started = time.monotonic()
    chunks: list[bytes] = []
    first_audio = None
    with session:
        connected = time.monotonic() - started
        session.synthesize(TTS_TEXT)
        for event in session:
            if event.type == "audio_chunk" and event.audio:
                if first_audio is None:
                    first_audio = time.monotonic() - started
                chunks.append(event.audio)
            elif event.type == "done":
                break
        sample_rate, audio_format = session.sample_rate, session.audio_format
    audio = b"".join(chunks)
    pcm16 = _f32_to_pcm16(audio) if audio_format == "pcm_f32" else audio
    _write_pcm16_wav(SPEECH_WAV, pcm16, sample_rate)
    print(
        f"  session ready in {connected:.2f}s, first audio at {first_audio:.2f}s, "
        f"{len(chunks)} chunks / {len(audio)} bytes ({audio_format}@{sample_rate})"
    )
    print(f"  saved speech to {SPEECH_WAV} (reused as STT input)")


def test_streaming_stt() -> None:
    url = _streaming_stt_url()
    print(f"  url: {url}")
    wav = _speech_wav()
    started = time.monotonic()
    first_delta = None
    parts: list[str] = []
    for delta in _client().asr.transcribe_stream(
        _wav_chunks(wav), language="ja", url=url
    ):
        if first_delta is None:
            first_delta = time.monotonic() - started
        parts.append(delta)
    print(
        f"  first delta at {first_delta:.2f}s, total {time.monotonic() - started:.2f}s"
        if first_delta is not None
        else f"  no transcript deltas (total {time.monotonic() - started:.2f}s)"
    )
    print(f"  transcript: {''.join(parts)!r}")


def test_stt_rest() -> None:
    url = _stt_rest_url()
    client = _client(url=url)
    probe = client.provider.http_probe
    print(f"  base url: {url}")
    print(f"  endpoint: POST {url}{client.provider.batch_transcribe_path}")
    print(f"  probe:    {url}{probe.probe_path} (deadline {probe.deadline_s:.0f}s)")
    wav = _speech_wav()
    started = time.monotonic()
    client.warmup()
    print(f"  warmup (readiness probe) done in {time.monotonic() - started:.2f}s")
    result = client.asr.transcribe(wav, language="ja")
    print(f"  transcribe done in {time.monotonic() - started:.2f}s")
    print(f"  transcript: {result.text!r}")


def test_s2st() -> None:
    url = _s2st_url()
    src = os.environ.get("FAL_S2ST_SRC", "en")
    tgt = os.environ.get("FAL_S2ST_TGT", "ja")
    print(f"  url: {url} ({src} -> {tgt})")
    source_file = os.environ.get("FAL_S2ST_INPUT")
    if source_file:
        wav = Path(source_file)
        print(f"  input: {wav}")
    else:
        wav = _speech_wav()
        print(
            f"  WARNING: no FAL_S2ST_INPUT set — the input may not be {src} "
            "speech; silence/empty output is EXPECTED then (plumbing check "
            f"only). Set FAL_S2ST_INPUT to a {src} speech file for a real test."
        )
    started = time.monotonic()
    result = _client().s2st.translate(wav, src=src, tgt=tgt, url=url)
    out = OUT_DIR / "s2st_out.wav"
    _write_pcm16_wav(out, result.audio, result.sample_rate)
    peak = max(
        (abs(s) for s in array.array("h", result.audio)), default=0
    )
    print(
        f"  done in {time.monotonic() - started:.2f}s, "
        f"{len(result.audio)} audio bytes (peak {peak}/32767) -> {out}"
    )
    print(f"  source transcript: {result.transcript_source!r}")
    if peak < 300 or not result.transcript_source:
        print(
            f"  NOTE: near-silent output / empty source transcript usually means "
            f"the model didn't hear {src} speech in the input, not an SDK error."
        )


TESTS = {
    "tts": test_tts,
    "streaming-stt": test_streaming_stt,
    "stt": test_stt_rest,
    "s2st": test_s2st,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        metavar=f"{{{','.join(TESTS)}}}",
        help="subset to run (default: all)",
    )
    tasks = parser.parse_args().tasks or list(TESTS)
    unknown = [t for t in tasks if t not in TESTS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {list(TESTS)}")

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY is not set", file=sys.stderr)
        return 2

    failures = 0
    for name in tasks:
        print(f"\n=== {name} ===")
        started = time.monotonic()
        try:
            TESTS[name]()
            print(f"--- {name}: OK ({time.monotonic() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — smoke test: report and continue
            failures += 1
            print(
                f"--- {name}: FAILED after {time.monotonic() - started:.1f}s "
                f"with {type(exc).__name__}: {exc}"
            )
    print(f"\n{len(tasks) - failures}/{len(tasks)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
