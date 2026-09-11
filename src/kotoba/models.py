"""Public pydantic v2 models returned by the SDK."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

# Accepts every wire name the TTS server echoes back in ``session.created``
# (see kotoba_sdk_gpu ``_OUTPUT_FORMAT_ALIASES``): a narrower Literal would make
# a negotiated response such as ``format="twilio"`` fail pydantic validation.
AudioFormat = Literal[
    "pcm16",
    "pcm_16",
    "pcm_s16le",
    "pcm_f32",
    "pcm_f32le",
    "float32",
    "mulaw",
    "ulaw",
    "twilio",
    "opus",
]


class TranscriptionStylePreference(BaseModel):
    """ASR transcription style preference.

    Mirrors the server's
    ``session.input_audio_transcription.style_preference`` shape.
    ``human_name="kana"`` transcribes personal names in katakana; ``None``
    (the default) lets the server decide.
    """

    human_name: Literal["kana"] | None = None

    def to_wire(self) -> dict[str, str]:
        """Flat field map shared by the WS and REST transports.

        Returns ``{}`` when no preference is set, so callers can merge it
        unconditionally.
        """

        return self.model_dump(exclude_none=True)

    @classmethod
    def coerce(
        cls, value: TranscriptionStylePreference | Mapping[str, Any] | None
    ) -> TranscriptionStylePreference | None:
        """Accept the JSON-shaped dict users naturally pass alongside the model."""
        if value is None or isinstance(value, cls):
            return value
        return cls.model_validate(value)


StreamEventType = Literal[
    "session_ready",
    "partial_transcript",
    "final_transcript",
    "audio_chunk",
    "committed",
    "error",
    "done",
]


class SessionConfig(BaseModel):
    """Per-session config passed into WebSocket ``stream()`` factories."""

    model_config = ConfigDict(extra="allow")

    sample_rate: int = 24000
    channels: int = 1
    audio_format: AudioFormat = "pcm16"
    src_language: str | None = None
    tgt_language: str | None = None
    speaker_id: str | None = None


class StreamEvent(BaseModel):
    """A single event yielded by a WebSocket streaming session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: StreamEventType
    text: str | None = None
    audio: bytes | None = None
    is_final: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobState(str, Enum):
    processing = "processing"
    done = "done"
    error = "error"


class JobIDResponse(BaseModel):
    """POST /transcription_jobs body."""

    job_id: str


class Segment(BaseModel):
    """Per-chunk timestamped transcript fragment."""

    text: str
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")


class JobStatus(BaseModel):
    """Combined view of GET /transcription_jobs/{id} outcomes.

    ``processing`` is an SDK-level synthesis of the server's HTTP 202 response.
    ``done`` carries ``transcription``; ``error`` carries ``error_message``.
    ``segments`` is populated only when the POST set ``with_timestamps=True``.
    """

    state: JobState
    transcription: str | None = None
    error_message: str | None = None
    segments: list[Segment] | None = None


class TranscriptResult(BaseModel):
    """Final result returned from the high-level ``transcribe`` helper."""

    model_config = ConfigDict(extra="allow")

    text: str
    job_id: str | None = None
    segments: list[Segment] | None = Field(
        default=None,
        description=(
            "Timed segments when with_timestamps=True and the deployment produced "
            "timings; None otherwise, in both transcribe() modes."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioResult(BaseModel):
    """Result returned by one-shot TTS / collected streaming runs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: bytes
    sample_rate: int
    audio_format: AudioFormat = "pcm16"
    content_type: str = "audio/pcm"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_wav(self, path: str) -> None:
        """Write the audio out as a playable WAV file."""

        from kotoba.audio import save_audio_as_wav

        save_audio_as_wav(path, self.data, self.sample_rate, self.audio_format)


class S2STResult(BaseModel):
    """Result returned from the speech-to-speech translation client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: bytes
    sample_rate: int
    audio_format: AudioFormat = "pcm16"
    transcript_source: str | None = None
    transcript_target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_wav(self, path: str) -> None:
        from kotoba.audio import save_audio_as_wav

        save_audio_as_wav(path, self.audio, self.sample_rate, self.audio_format)
