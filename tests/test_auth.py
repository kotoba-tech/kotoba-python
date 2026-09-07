"""Auth-scheme selection from the endpoint URL."""

from __future__ import annotations

import pytest
from kotoba._auth import auth_headers, is_fal_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://fal.run/team/app", True),
        ("wss://fal.run/team/app/v1/realtime", True),
        ("wss://queue.fal.run/team/app", True),
        ("https://notfal.run/team/app", False),
        ("https://fal.run.evil.example/x", False),
        ("wss://asr.example.com/v1/realtime", False),
        ("", False),
        (None, False),
    ],
)
def test_is_fal_url(url, expected):
    assert is_fal_url(url) is expected


def test_auth_headers_scheme_follows_host():
    assert auth_headers("tok", "https://api.example.com/v1") == {"Authorization": "Bearer tok"}
    assert auth_headers("tok", "wss://fal.run/team/app/v2/tts/ws") == {"Authorization": "Key tok"}


def test_auth_headers_without_key():
    assert auth_headers(None, "https://fal.run/team/app") == {}
    assert auth_headers("", "https://api.example.com") == {}
