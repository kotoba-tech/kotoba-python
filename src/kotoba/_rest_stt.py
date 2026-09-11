"""Wire contract of the one-shot ``POST /v1/speech-to-text`` endpoint.

The ASR clients in ``asr.py`` only orchestrate: build the request, post the
file, parse the payload. Everything the endpoint dictates lives here: the form
field names and encodings and the shape of its response. Same split as the WebSocket protocol modules (``_ws_asr.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from kotoba.errors import ProtocolError
from kotoba.models import Segment, TranscriptionStylePreference, TranscriptResult

# Relative to the session base (``KOTOBA_ASR_REST_URL``, which includes ``/v1``),
# exactly like the job API's ``/transcription_jobs``.
SPEECH_TO_TEXT_PATH = "/speech-to-text"

FileFormat = Literal["other", "pcm_s16le_16"]


@dataclass(frozen=True)
class SpeechToTextRequest:
    """Form fields of one ``POST /v1/speech-to-text``."""

    form: dict[str, Any]
    path: str = SPEECH_TO_TEXT_PATH

    @classmethod
    def build(
        cls,
        *,
        language: str | None,
        keywords: list[str] | None,
        style_preference: TranscriptionStylePreference | dict | None,
        with_timestamps: bool,
        file_format: FileFormat,
    ) -> SpeechToTextRequest:
        # ``language`` is omitted when None so the server applies its configured
        # default; ``keywords`` goes out as repeated fields; the SDK-wide
        # ``style_preference`` maps onto this endpoint's ``kana`` flag.
        form: dict[str, Any] = {
            "file_format": file_format,
            "with_timestamps": str(with_timestamps).lower(),
        }
        if language is not None:
            form["language"] = language
        if keywords:
            form["keywords"] = list(keywords)
        preference = TranscriptionStylePreference.coerce(style_preference)
        if preference is not None and preference.human_name == "kana":
            form["kana"] = "true"
        # Auth is the session's ``Authorization`` header (``Bearer``, or ``Key``
        # on fal hosts, where the gateway authenticates the caller).
        return cls(form=form)


class _Word(BaseModel):
    """Mirror of the server's ``SpeechToTextWord``; unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    text: str
    start: float | None = None
    end: float | None = None


class _SpeechToTextResponse(BaseModel):
    """Mirror of the server's ``SpeechToTextResponse``.

    Source: ``pydantic_models/stt/speech_to_text.py``. Required fields are
    required here too, so a deployment that stops honouring the contract fails
    on the first call instead of yielding an empty transcript.
    """

    model_config = ConfigDict(extra="ignore")

    language_code: str
    language_probability: float
    text: str
    words: list[_Word]
    audio_duration_secs: float | None = None


def parse_speech_to_text(payload: Any) -> TranscriptResult:
    """Map the endpoint's 2xx body onto the SDK-wide :class:`TranscriptResult`.

    Raises :class:`ProtocolError` (``status_code=200``) when the body does not
    match the server contract.
    """
    try:
        body = _SpeechToTextResponse.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise ProtocolError(
            f"Malformed /speech-to-text response: {where}: {first['msg']}",
            status_code=200,
            payload=payload if isinstance(payload, dict) else {"detail": str(payload)[:512]},
        ) from exc

    # words=[] unless timestamps were requested; an entry without timing
    # (spacing, audio events) carries no start/end.
    timed = [w for w in body.words if w.start is not None and w.end is not None]
    segments = [Segment(text=w.text, start=w.start, end=w.end) for w in timed] or None
    metadata: dict[str, Any] = {"language_code": body.language_code, "language_probability": body.language_probability}
    if body.audio_duration_secs is not None:
        metadata["audio_duration_secs"] = body.audio_duration_secs
    return TranscriptResult(text=body.text, segments=segments, metadata=metadata)
