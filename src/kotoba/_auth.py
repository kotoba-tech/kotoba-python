"""Authorization header selection.

The Kotoba services are reachable directly or through fal.ai's gateway.
The only client-visible difference is the auth scheme: fal expects
``Authorization: Key <api_key>``, the direct endpoints ``Bearer``. The
scheme is derived from the endpoint host so callers configure nothing.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def is_fal_url(url: str | None) -> bool:
    if not url:
        return False
    host = urlsplit(url).hostname
    return host is not None and (host == "fal.run" or host.endswith(".fal.run"))


def auth_headers(api_key: str | None, url: str) -> dict[str, str]:
    if not api_key:
        return {}
    scheme = "Key" if is_fal_url(url) else "Bearer"
    return {"Authorization": f"{scheme} {api_key}"}
