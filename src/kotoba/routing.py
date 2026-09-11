"""Endpoint routing table.

Single source of truth for `(modality, src, tgt) -> wss://...` URLs. The
registry is seeded from environment variables at import time; callers can
also add or override routes at runtime via `register_endpoint`.

`(modality, None, None)` is the service default, used for every language:
the language is chosen per session (TTS ``open``, S2ST session update), and a
deployment serves whichever languages it was configured with. Register a
language-specific route only when that language is served by a different
deployment.
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
    ("asr", None, None, "KOTOBA_ASR_URL"),
    ("tts", None, None, "KOTOBA_TTS_URL"),
    ("s2st", None, None, "KOTOBA_S2ST_URL"),
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
    """Look up the WebSocket URL for a route.

    The exact ``(modality, src, tgt)`` entry wins; otherwise the service
    default ``(modality, None, None)`` is used.
    """

    for key in ((modality, src, tgt), (modality, None, None)):
        if key in _REGISTRY:
            return _REGISTRY[key]
    registered = sorted(f"{m}:{s}->{t}" for (m, s, t) in _REGISTRY)
    raise UnsupportedRouteError(
        f"No endpoint registered for {modality} {src!r} -> {tgt!r}. "
        f"Registered routes: {registered or '(none)'}. "
        f"Set KOTOBA_{modality.upper()}_URL, pass the URL to KotobaClient, "
        f"or call kotoba.register_endpoint(...)."
    )


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
