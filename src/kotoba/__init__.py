"""Kotoba Speech SDK — REST transcription jobs + WebSocket streaming (ASR, TTS, S2ST)."""

from kotoba._version import __version__
from kotoba.asr import ASRClient, AsyncASRClient
from kotoba.client import AsyncKotobaClient, KotobaClient
from kotoba.errors import (
    APIError,
    AuthError,
    JobNotFoundError,
    KotobaError,
    ProtocolError,
    TimeoutError,
    TranscriptionError,
    UnsupportedRouteError,
)
from kotoba.models import (
    AudioResult,
    JobIDResponse,
    JobState,
    JobStatus,
    S2STResult,
    Segment,
    SessionConfig,
    StreamEvent,
    TextSource,
    TranscriptResult,
)
from kotoba.routing import endpoint_for, register_endpoint
from kotoba.s2st import AsyncS2STClient, S2STClient
from kotoba.tts import AsyncTTSClient, TTSClient

__all__ = [
    "__version__",
    # Facades
    "KotobaClient",
    "AsyncKotobaClient",
    # Per-modality clients
    "ASRClient",
    "AsyncASRClient",
    "TTSClient",
    "AsyncTTSClient",
    "S2STClient",
    "AsyncS2STClient",
    # Data models
    "JobIDResponse",
    "JobState",
    "JobStatus",
    "Segment",
    "TranscriptResult",
    "AudioResult",
    "S2STResult",
    "SessionConfig",
    "StreamEvent",
    "TextSource",
    # Errors
    "KotobaError",
    "APIError",
    "AuthError",
    "ProtocolError",
    "TimeoutError",
    "JobNotFoundError",
    "TranscriptionError",
    "UnsupportedRouteError",
    # Routing
    "endpoint_for",
    "register_endpoint",
]
