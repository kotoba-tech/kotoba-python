import pytest
from kotoba.models import (
    AudioResult,
    S2STResult,
    SessionConfig,
    StreamEvent,
    TranscriptionStylePreference,
)
from pydantic import ValidationError


def test_session_config_defaults():
    cfg = SessionConfig()
    assert cfg.sample_rate == 24000
    assert cfg.audio_format == "pcm16"


def test_stream_event_audio_chunk():
    ev = StreamEvent(type="audio_chunk", audio=b"\x00\x01")
    assert ev.audio == b"\x00\x01"
    assert ev.text is None


def test_stream_event_invalid_type_rejected():
    with pytest.raises(ValidationError):
        StreamEvent(type="not_a_real_event")  # type: ignore[arg-type]


def test_audio_result_to_wav_pcm16(tmp_path):
    samples = (b"\x00\x10" * 100)  # ~200 bytes pcm16
    out = tmp_path / "out.wav"
    AudioResult(data=samples, sample_rate=16000, audio_format="pcm16").to_wav(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_audio_result_to_wav_pcm_f32(tmp_path):
    import struct

    samples = b"".join(struct.pack("<f", 0.1) for _ in range(100))
    out = tmp_path / "out.wav"
    AudioResult(data=samples, sample_rate=24000, audio_format="pcm_f32").to_wav(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_s2st_result_minimal():
    r = S2STResult(audio=b"", sample_rate=24000, audio_format="pcm16")
    assert r.transcript_source is None


@pytest.mark.parametrize(
    "audio_format",
    [
        "pcm16",
        "pcm_16",
        "pcm_s16le",
        "pcm_f32",
        "pcm_f32le",
        "float32",
        "mulaw",
        "ulaw",
        "twilio",
        "opus",
    ],
)
def test_audio_result_accepts_negotiated_formats(audio_format):
    r = AudioResult(data=b"", sample_rate=24000, audio_format=audio_format)
    assert r.audio_format == audio_format


def test_audio_result_rejects_unknown_format():
    with pytest.raises(ValidationError):
        AudioResult(data=b"", sample_rate=24000, audio_format="flac")  # type: ignore[arg-type]


def test_style_preference_coerce_none():
    assert TranscriptionStylePreference.coerce(None) is None


def test_style_preference_coerce_instance_passthrough():
    pref = TranscriptionStylePreference(human_name="kana")
    assert TranscriptionStylePreference.coerce(pref) is pref


def test_style_preference_coerce_dict():
    coerced = TranscriptionStylePreference.coerce({"human_name": "kana"})
    assert coerced == TranscriptionStylePreference(human_name="kana")


def test_style_preference_coerce_rejects_invalid_dict():
    with pytest.raises(ValidationError):
        TranscriptionStylePreference.coerce({"human_name": "unknown"})
