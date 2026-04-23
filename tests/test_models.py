import pytest
from pydantic import ValidationError

from kotoba.models import AudioResult, S2STResult, SessionConfig, StreamEvent


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
