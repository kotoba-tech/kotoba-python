"""Live release check of the streaming ASR surface against real deployments.

Not part of the package and not run by pytest: it needs a running
``streaming_stt`` deployment and sends real audio at real-time pace. Run it
before a release with the SDK under test installed (the built wheel, or the
source tree via ``uv run``):

    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_URL=wss://.../v1/realtime          # serves diarization
    export KOTOBA_ASR_URL_NO_DIARIZATION=wss://.../v1/realtime  # optional, for (d)
    uv run scripts/live_check.py --keywords 東京 --keywords 大阪

The default clip is Japanese. For another language the deployments serve, give
its code and a clip in that language (with keywords that occur in the clip):

    uv run scripts/live_check.py --language en --audio clip_en.wav --keywords Tokyo

Checks (one row each in the pass/fail table):

  a. ``stream(diarization=True, timestamps=True)``: every non-blank partial
     and every final carries ``speaker_id``; every final has ``start_ms`` /
     ``end_ms``.
  b. ``transcribe_stream(diarization=True, keywords=[...])``: yields
     ``StreamEvent`` with speakers, and the keywords appear in the text.
  c. ``transcribe_stream()`` with defaults: yields ``str`` and the session
     update carries no ``diarization`` / ``turn_detection`` field.
  d. ``diarization=True`` on a deployment without diarization raises
     ``ProtocolError(code="invalid_parameters", param="session.diarization")``.
  e. A ``--long-seconds`` (default 120 s) diarized session at real-time pace
     ends with ``committed`` and no handshake or idle timeout.
  f. A language code the deployment does not serve (``--unsupported-language``)
     raises ``ProtocolError(code="invalid_parameters",
     param="session.input_audio_transcription.target_language")``.

Checks a-e pass ``--language`` (default ``ja``) to the SDK. Each deployment is
warmed up first (a plain session in that language, retried until
``--warmup-seconds``), so a scaled-to-zero app does not fail the handshake.
The exit status is non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import kotoba
from kotoba._ws_asr import AsyncASRSession
from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16

DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "examples" / "audio" / "ja" / "example.mp3"
SAMPLE_RATE = 24000
CHUNK_MS = 100
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * 2

# Every session update the SDK sends, as (url, frame), so (c) can inspect the
# wire request the default path builds.
SENT_UPDATES: list[tuple[str, dict[str, Any]]] = []
_send_json = AsyncASRSession._send_json


async def _recording_send_json(self: AsyncASRSession, payload: dict[str, Any]) -> None:
    if payload.get("type") == "transcription_session.update":
        SENT_UPDATES.append((self._url, copy.deepcopy(payload)))
    await _send_json(self, payload)


AsyncASRSession._send_json = _recording_send_json


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def load_pcm16(path: Path, seconds: float) -> bytes:
    """Load ``path`` as 24 kHz mono pcm16, looped or cut to ``seconds``."""

    audio, rate = load_mono_pcm16_wav(path)
    if rate != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, rate, SAMPLE_RATE)
    pcm = audio.astype("<i2").tobytes()
    want = int(seconds * SAMPLE_RATE) * 2
    while len(pcm) < want:
        pcm += pcm
    return pcm[:want]


def _offsets(pcm: bytes) -> range:
    return range(0, len(pcm), CHUNK_BYTES)


async def paced(pcm: bytes) -> AsyncIterator[bytes]:
    """Yield pcm16 chunks at wall-clock pace, like a live microphone."""

    start = time.monotonic()
    for i, offset in enumerate(_offsets(pcm)):
        await asyncio.sleep(max(0.0, start + i * CHUNK_MS / 1000 - time.monotonic()))
        yield pcm[offset : offset + CHUNK_BYTES]


def paced_sync(pcm: bytes) -> Iterator[bytes]:
    start = time.monotonic()
    for i, offset in enumerate(_offsets(pcm)):
        time.sleep(max(0.0, start + i * CHUNK_MS / 1000 - time.monotonic()))
        yield pcm[offset : offset + CHUNK_BYTES]


def _speaker_problems(events: list[kotoba.StreamEvent], *, need_span: bool) -> list[str]:
    problems = []
    partials = [e for e in events if e.type == "partial_transcript" and e.text and e.text.strip()]
    finals = [e for e in events if e.type == "final_transcript"]
    if not partials:
        problems.append("no non-blank partial")
    if not finals:
        problems.append("no final")
    missing = sum(1 for e in partials if not e.speaker_id)
    if missing:
        problems.append(f"{missing}/{len(partials)} non-blank partials without speaker_id")
    missing = sum(1 for e in finals if not e.speaker_id)
    if missing:
        problems.append(f"{missing}/{len(finals)} finals without speaker_id")
    if need_span:
        missing = sum(1 for e in finals if e.metadata.get("start_ms") is None or e.metadata.get("end_ms") is None)
        if missing:
            problems.append(f"{missing}/{len(finals)} finals without start_ms/end_ms")
    return problems


def _speaker_summary(events: list[kotoba.StreamEvent]) -> dict[str, Any]:
    partials = [e for e in events if e.type == "partial_transcript" and e.text and e.text.strip()]
    finals = [e for e in events if e.type == "final_transcript"]
    return {
        "partials": len(partials),
        "finals": len(finals),
        "speakers": sorted({e.speaker_id for e in partials + finals if e.speaker_id}),
        "text": "".join(e.text or "" for e in finals),
    }


async def warm_up(url: str, deadline_s: float, language: str) -> str:
    """Open plain sessions until one is acknowledged; return a short note."""

    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            async with client.asr.stream(language=language, url=url):
                return f"ready after {time.monotonic() - start:.1f} s, {attempts} attempt(s)"
        except (kotoba.APIError, OSError) as exc:
            # A rejected session option (a --language the deployment does not serve) never becomes ready.
            if isinstance(exc, kotoba.ProtocolError) and exc.code == "invalid_parameters":
                raise
            if time.monotonic() - start > deadline_s:
                raise RuntimeError(f"not ready after {deadline_s:.0f} s: {exc}") from exc
            await asyncio.sleep(5)


async def check_stream(url: str, pcm: bytes, language: str) -> Result:
    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    events: list[kotoba.StreamEvent] = []
    ack = None
    async with client.asr.stream(language=language, diarization=True, timestamps=True, url=url) as session:
        feeder = asyncio.create_task(session.feed(paced(pcm)))
        async for event in session:
            if event.type == "session_ready":
                ack = (event.metadata.get("session") or {}).get("diarization")
            if event.type == "done":
                break
            events.append(event)
        await feeder
    problems = _speaker_problems(events, need_span=True)
    summary = _speaker_summary(events)
    detail = "; ".join(problems) or (
        f"{summary['partials']} non-blank partials and {summary['finals']} finals, all with a speaker "
        f"({', '.join(summary['speakers'])}); finals timed; ack diarization={ack}"
    )
    return Result("", not problems, detail, extra=summary)


async def check_keywords(url: str, pcm: bytes, language: str, keywords: list[str]) -> Result:
    if not keywords:
        return Result("", False, "no --keywords given")
    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    events = [
        event
        async for event in client.asr.transcribe_stream(
            paced(pcm), language=language, diarization=True, keywords=keywords, url=url
        )
    ]
    problems = []
    if not all(isinstance(e, kotoba.StreamEvent) for e in events):
        problems.append("yielded something other than StreamEvent")
    problems += _speaker_problems(events, need_span=False)
    summary = _speaker_summary(events)
    text = summary["text"] or "".join(e.text or "" for e in events if e.type == "partial_transcript")
    hits = [k for k in keywords if k in text]
    if len(hits) != len(keywords):
        problems.append(f"keywords missing from the text: {[k for k in keywords if k not in hits]}")
    sent = SENT_UPDATES[-1][1]["session"]["input_audio_transcription"].get("keywords")
    if sent != keywords:
        problems.append(f"session update sent keywords={sent}")
    detail = "; ".join(problems) or (
        f"keyword hits {len(hits)}/{len(keywords)} ({', '.join(hits)}); "
        f"{summary['partials']} partials and {summary['finals']} finals with a speaker"
    )
    return Result("", not problems, detail, extra=summary)


def check_default(url: str, pcm: bytes, language: str) -> Result:
    """The default path, through the sync client."""

    client = kotoba.KotobaClient(asr_ws_url=url)
    sent_before = len(SENT_UPDATES)
    deltas = list(client.asr.transcribe_stream(paced_sync(pcm), language=language, url=url))
    problems = []
    if not deltas:
        problems.append("no delta")
    if not all(isinstance(d, str) for d in deltas):
        problems.append(f"yielded {sorted({type(d).__name__ for d in deltas})}, not only str")
    updates = [frame for _, frame in SENT_UPDATES[sent_before:]]
    if len(updates) != 1:
        problems.append(f"{len(updates)} session updates sent")
    else:
        session = updates[0]["session"]
        for key in ("diarization", "turn_detection"):
            if key in session:
                problems.append(f"session update has {key}={session[key]!r}")
    detail = "; ".join(problems) or (
        f"{len(deltas)} str deltas; session update keys: {sorted(updates[0]['session'])}"
    )
    return Result("", not problems, detail, extra={"text": "".join(deltas)})


async def check_no_diarization(url: str, pcm: bytes, language: str) -> Result:
    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    try:
        async with client.asr.stream(language=language, diarization=True, url=url) as session:
            await session.send_audio(pcm[: CHUNK_BYTES * 10])
            await session.commit()
            async for event in session:
                if event.type == "done":
                    break
    except kotoba.ProtocolError as exc:
        ok = exc.code == "invalid_parameters" and exc.param == "session.diarization"
        source = "error frame" if exc.payload.get("type") == "error" else "acknowledgement without diarization"
        return Result("", ok, f"ProtocolError code={exc.code} param={exc.param} (from the {source})")
    return Result("", False, "session ran without an error")


async def check_long(url: str, pcm: bytes, language: str) -> Result:
    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    start = time.monotonic()
    arrivals: list[float] = []
    events: list[kotoba.StreamEvent] = []
    # transcribe_stream() returns only after the server's `committed`; a
    # handshake or idle timeout raises instead.
    async for event in client.asr.transcribe_stream(paced(pcm), language=language, diarization=True, url=url):
        events.append(event)
        if event.text and event.text.strip():
            arrivals.append(time.monotonic() - start)
    elapsed = time.monotonic() - start
    audio_s = len(pcm) / 2 / SAMPLE_RATE
    gaps = [b - a for a, b in zip([0.0] + arrivals, arrivals)]
    problems = _speaker_problems(events, need_span=False)
    if elapsed < audio_s:
        problems.append(f"ended after {elapsed:.1f} s, before the {audio_s:.0f} s of audio")
    if not arrivals or arrivals[-1] < audio_s - 20:
        problems.append(f"last text at {arrivals[-1] if arrivals else 0:.1f} s of {audio_s:.0f} s")
    detail = "; ".join(problems) or (
        f"{audio_s:.0f} s of audio, session {elapsed:.1f} s, last text at {arrivals[-1]:.1f} s, "
        f"longest text gap {max(gaps):.1f} s, {len(events)} events"
    )
    return Result("", not problems, detail)


async def check_unsupported_language(url: str, language: str) -> Result:
    client = kotoba.AsyncKotobaClient(asr_ws_url=url)
    try:
        # The server refuses the session update, so entering the session raises.
        async with client.asr.stream(language=language, url=url):
            pass
    except kotoba.ProtocolError as exc:
        ok = exc.code == "invalid_parameters" and exc.param == "session.input_audio_transcription.target_language"
        return Result("", ok, f"ProtocolError code={exc.code} param={exc.param} for language={language!r}")
    return Result("", False, f"the deployment acknowledged a session with language={language!r}")


async def run_check(name: str, check, *args) -> Result:
    start = time.monotonic()
    try:
        if asyncio.iscoroutinefunction(check):
            result = await check(*args)
        else:
            result = await asyncio.to_thread(check, *args)
    except Exception as exc:  # report it and go on with the other checks
        result = Result("", False, f"{type(exc).__name__}: {exc}")
    result.name = name
    result.seconds = time.monotonic() - start
    print(f"  {name}: {'PASS' if result.ok else 'FAIL'}", flush=True)
    return result


async def main(args: argparse.Namespace) -> int:
    url = args.url or os.environ.get("KOTOBA_ASR_URL")
    no_diar_url = args.no_diarization_url or os.environ.get("KOTOBA_ASR_URL_NO_DIARIZATION")
    if not url:
        sys.exit("set KOTOBA_ASR_URL or pass --url")
    clip = load_pcm16(args.audio, args.clip_seconds)
    long = load_pcm16(args.long_audio or args.audio, args.long_seconds)
    language = args.language
    print(f"kotoba-sdk {kotoba.__version__} from {Path(kotoba.__file__).parent}; language {language}")

    for target in [url] + ([no_diar_url] if no_diar_url else []):
        print(f"warm-up {target}: {await warm_up(target, args.warmup_seconds, language)}", flush=True)
    results = [
        await run_check("a. stream(diarization, timestamps)", check_stream, url, clip, language),
        await run_check(
            "b. transcribe_stream(diarization, keywords)", check_keywords, url, clip, language, args.keywords
        ),
        await run_check("c. transcribe_stream() defaults", check_default, url, clip, language),
    ]
    name = "d. diarization on a deployment without it"
    if no_diar_url:
        results.append(await run_check(name, check_no_diarization, no_diar_url, clip, language))
    else:
        results.append(Result(name, False, "skipped: no KOTOBA_ASR_URL_NO_DIARIZATION"))
    results.append(await run_check(f"e. {args.long_seconds:.0f} s diarized session", check_long, url, long, language))
    results.append(
        await run_check(
            f"f. unsupported language {args.unsupported_language!r}",
            check_unsupported_language,
            url,
            args.unsupported_language,
        )
    )

    print("\n| Check | Result | Time | Detail |\n|---|---|---|---|")
    for r in results:
        print(f"| {r.name} | {'PASS' if r.ok else 'FAIL'} | {r.seconds:.1f} s | {r.detail} |")
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="diarizing deployment (default: KOTOBA_ASR_URL)")
    parser.add_argument(
        "--no-diarization-url", help="deployment without diarization (default: KOTOBA_ASR_URL_NO_DIARIZATION)"
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="language code checks a-e pass to the SDK; both deployments must serve it, and --audio / "
        "--long-audio / --keywords must be in that language (default: ja, the language of the default clip)",
    )
    parser.add_argument(
        "--unsupported-language", default="xx", help="code the deployment does not serve, for check f (default: xx)"
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="clip for checks a-d")
    parser.add_argument("--clip-seconds", type=float, default=30.0, help="clip length, looped if shorter")
    parser.add_argument("--long-audio", type=Path, help="audio for check e (default: --audio, looped)")
    parser.add_argument("--long-seconds", type=float, default=120.0)
    parser.add_argument("--keywords", action="append", default=[], help="term that occurs in the clip; repeat")
    parser.add_argument("--warmup-seconds", type=float, default=300.0)
    parser.add_argument("--json", type=Path, help="also write the results here")
    sys.exit(asyncio.run(main(parser.parse_args())))
