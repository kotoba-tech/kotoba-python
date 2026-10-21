"""Public pydantic v2 models returned by the SDK."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictInt

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


class ServerVAD(BaseModel):
    """Blank-model-output turn detection configuration for streaming ASR."""

    type: Literal["server_vad"] = "server_vad"
    silence_duration_ms: StrictInt | None = Field(default=800, ge=0)


def normalize_keywords(keywords: list[str] | None) -> list[str] | None:
    """Strip the terms and drop blank ones; ``None`` or an all-blank list means no keywords.

    The count and length limits belong to the deployment and are enforced by the
    server (it answers ``invalid_parameters``), so a server-side change needs no
    SDK release. A bare string is refused here: it would otherwise be sent as a
    list of characters.
    """

    if keywords is None:
        return None
    if isinstance(keywords, str) or not all(isinstance(term, str) for term in keywords):
        raise ValueError("keywords must be a list of strings")
    terms = [term.strip() for term in keywords if term.strip()]
    return terms or None


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


class VoiceSettings(TypedDict, total=False):
    """TTS ``voice_settings``: ``speed`` and ``volume`` multipliers, ``pitch`` in semitones.

    For static checking only; the server owns the ranges and rejects values outside them.
    """

    speed: float
    pitch: float
    volume: float


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
    speaker_id: str | None = Field(
        default=None,
        description="Speaker label when the request asked for diarization and the model attributed the word; None otherwise.",
    )


class Utterance(BaseModel):
    """A run of consecutive words attributed to one speaker (``transcribe(diarization=True)``)."""

    speaker_id: str | None = Field(description="Speaker label, or None where the model did not attribute the words.")
    text: str
    start: float | None = Field(default=None, description="Start of the first timed word, when with_timestamps=True.")
    end: float | None = Field(default=None, description="End of the last timed word, when with_timestamps=True.")


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
    utterances: list[Utterance] | None = Field(
        default=None,
        description=(
            "Runs of consecutive words by speaker, in reading order, when diarization=True "
            "on the one-shot endpoint; None otherwise. Times are set only with with_timestamps=True."
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
