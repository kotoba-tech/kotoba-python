"""Kotoba Speech SDK — REST transcription jobs + WebSocket streaming (ASR, TTS, S2ST)."""

from kotoba._providers import FAL, KOTOBA, ProbePolicy, ProviderConfig, RetryPolicy
from kotoba._version import __version__
from kotoba.asr import ASRClient, AsyncASRClient
from kotoba.client import AsyncKotobaClient, KotobaClient
from kotoba.errors import (
    APIError,
    AuthError,
    KotobaError,
    ProtocolError,
    TimeoutError,
    UnsupportedRouteError,
    WorkerStartupError,
)
from kotoba.models import (
    AudioResult,
    S2STResult,
    Segment,
    SessionConfig,
    StreamEvent,
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
    "Segment",
    "TranscriptResult",
    "AudioResult",
    "S2STResult",
    "SessionConfig",
    "StreamEvent",
    # Providers
    "ProviderConfig",
    "RetryPolicy",
    "ProbePolicy",
    "KOTOBA",
    "FAL",
    # Errors
    "KotobaError",
    "APIError",
    "AuthError",
    "ProtocolError",
    "TimeoutError",
    "WorkerStartupError",
    "UnsupportedRouteError",
    # Routing
    "endpoint_for",
    "register_endpoint",
]
