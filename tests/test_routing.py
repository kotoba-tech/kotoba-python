import pytest

from kotoba import endpoint_for, register_endpoint
from kotoba.errors import UnsupportedRouteError
from kotoba.routing import _seed_default_routes


def test_env_var_seeds_route(monkeypatch):
    monkeypatch.setenv("KOTOBA_S2ST_EN_JA_URL", "wss://env.test/s2st")
    _seed_default_routes()
    assert endpoint_for("s2st", "en", "ja") == "wss://env.test/s2st"


def test_register_then_lookup():
    register_endpoint("tts", None, "ja", "wss://example.test/ja-tts")
    assert endpoint_for("tts", None, "ja") == "wss://example.test/ja-tts"


def test_unsupported_route_raises_with_helpful_message():
    with pytest.raises(UnsupportedRouteError) as exc:
        endpoint_for("s2st", "fr", "de")
    assert "fr" in str(exc.value)
    assert "register_endpoint" in str(exc.value)


def test_register_replaces_existing():
    register_endpoint("tts", None, "ko", "wss://a.test")
    register_endpoint("tts", None, "ko", "wss://b.test")
    assert endpoint_for("tts", None, "ko") == "wss://b.test"
