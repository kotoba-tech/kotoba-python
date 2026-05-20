"""Public pydantic v2 models returned by the SDK."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AudioFormat = Literal["pcm16", "pcm_f32"]

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
        description="Populated only when transcribe(..., with_timestamps=True).",
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
