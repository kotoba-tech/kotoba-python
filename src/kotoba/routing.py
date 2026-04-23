"""Endpoint routing table.

Single source of truth for `(modality, src, tgt) -> wss://...` URLs. The
registry is seeded from environment variables at import time; callers can
also add or override routes at runtime via `register_endpoint`.
"""

from __future__ import annotations

import os
from typing import Literal

from kotoba.errors import UnsupportedRouteError

Modality = Literal["asr", "tts", "s2st"]
RouteKey = tuple[Modality, str | None, str | None]


_REGISTRY: dict[RouteKey, str] = {}


# Known routes that can be seeded from the environment. Each entry is
# `(modality, src, tgt, env_var_name)`. To expose staging / prod URLs to
# the SDK, export the corresponding variable before import.
_ENV_SEEDED_ROUTES: tuple[tuple[Modality, str | None, str | None, str], ...] = (
    ("s2st", "en", "ja", "KOTOBA_S2ST_EN_JA_URL"),
    ("tts", None, "ja", "KOTOBA_TTS_JA_URL"),
    ("asr", None, None, "KOTOBA_ASR_URL"),
)


def register_endpoint(
    modality: Modality,
    src: str | None,
    tgt: str | None,
    url: str,
) -> None:
    """Register or replace a `(modality, src, tgt) -> url` mapping."""

    _REGISTRY[(modality, src, tgt)] = url


def endpoint_for(modality: Modality, src: str | None, tgt: str | None) -> str:
    """Look up the WebSocket URL for a given route."""

    key: RouteKey = (modality, src, tgt)
    if key not in _REGISTRY:
        registered = sorted(f"{m}:{s}->{t}" for (m, s, t) in _REGISTRY)
        raise UnsupportedRouteError(
            f"No endpoint registered for {modality} {src!r} -> {tgt!r}. "
            f"Registered routes: {registered or '(none)'}. "
            f"Call kotoba.register_endpoint(...) to add one, or set the "
            f"corresponding KOTOBA_*_URL environment variable."
        )
    return _REGISTRY[key]


def registered_routes() -> list[RouteKey]:
    """Return all currently registered routes (testing / debugging)."""

    return list(_REGISTRY.keys())


def _seed_default_routes() -> None:
    """Seed the registry from environment variables.

    Only routes whose env var is set get registered; everything else
    stays absent until the caller configures it explicitly.
    """

    for modality, src, tgt, env_var in _ENV_SEEDED_ROUTES:
        url = os.environ.get(env_var)
        if url:
            register_endpoint(modality, src, tgt, url)


_seed_default_routes()
