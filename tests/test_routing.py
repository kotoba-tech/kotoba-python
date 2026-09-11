import pytest
from kotoba import endpoint_for, register_endpoint
from kotoba.errors import UnsupportedRouteError
from kotoba.routing import _REGISTRY, _seed_default_routes


@pytest.fixture(autouse=True)
def _isolated_registry():
    """The registry is a module global; keep each test's routes to itself."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def test_env_var_seeds_the_service_default(monkeypatch):
    monkeypatch.setenv("KOTOBA_S2ST_URL", "wss://env.test/s2st")
    _seed_default_routes()
    # The default serves every language pair; the pair is chosen per session.
    assert endpoint_for("s2st", "en", "ja") == "wss://env.test/s2st"
    assert endpoint_for("s2st", "ja", "en") == "wss://env.test/s2st"


def test_language_specific_route_beats_the_service_default():
    register_endpoint("tts", None, None, "wss://default.test/tts")
    register_endpoint("tts", None, "en", "wss://en-only.test/tts")
    assert endpoint_for("tts", None, "en") == "wss://en-only.test/tts"
    assert endpoint_for("tts", None, "ja") == "wss://default.test/tts"


def test_unsupported_route_when_no_default():
    with pytest.raises(UnsupportedRouteError) as exc:
        endpoint_for("tts", None, "ko")
    assert "KOTOBA_TTS_URL" in str(exc.value)


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
