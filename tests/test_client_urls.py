"""KotobaClient URL kwargs register the per-service WebSocket defaults."""

import kotoba
import pytest
from kotoba.routing import _REGISTRY


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    for var in ("KOTOBA_API_KEY", "KOTOBA_ASR_REST_URL", "KOTOBA_ASR_URL", "KOTOBA_TTS_URL", "KOTOBA_S2ST_URL"):
        monkeypatch.delenv(var, raising=False)
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


@pytest.mark.parametrize("client_class", [kotoba.KotobaClient, kotoba.AsyncKotobaClient])
def test_explicit_urls_register_the_service_defaults(client_class):
    client = client_class(api_key="k", asr_ws_url="wss://a.test", tts_ws_url="wss://t.test", s2st_ws_url="wss://s.test")

    assert (client.asr_ws_url, client.tts_ws_url, client.s2st_ws_url) == (
        "wss://a.test",
        "wss://t.test",
        "wss://s.test",
    )
    # Any language resolves to the service default; the language travels in the session.
    assert kotoba.endpoint_for("tts", None, "ja") == "wss://t.test"
    assert kotoba.endpoint_for("tts", None, "en") == "wss://t.test"
    assert kotoba.endpoint_for("s2st", "en", "ja") == "wss://s.test"
    assert kotoba.endpoint_for("asr", None, None) == "wss://a.test"


def test_env_vars_fill_in_missing_urls(monkeypatch):
    monkeypatch.setenv("KOTOBA_TTS_URL", "wss://env.test/tts")
    monkeypatch.setenv("KOTOBA_S2ST_URL", "wss://env.test/sts")

    client = kotoba.KotobaClient(api_key="k")

    assert client.tts_ws_url == "wss://env.test/tts"
    assert client.s2st_ws_url == "wss://env.test/sts"
    assert kotoba.endpoint_for("s2st", "ja", "en") == "wss://env.test/sts"


def test_language_specific_kwargs_are_gone():
    with pytest.raises(TypeError):
        kotoba.KotobaClient(api_key="k", tts_ja_ws_url="wss://t.test")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        kotoba.AsyncKotobaClient(api_key="k", s2st_en_ja_ws_url="wss://s.test")  # type: ignore[call-arg]
