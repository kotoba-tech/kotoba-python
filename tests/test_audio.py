import struct

import numpy as np
import pytest

from kotoba.audio import (
    load_mono_pcm16_wav,
    mulaw_bytes_to_int16,
    pcm_f32_bytes_to_int16,
    resample_mono_pcm16,
    save_audio_as_wav,
    save_mono_pcm16_wav,
    save_pcm_f32_as_wav,
)


def test_save_then_load_roundtrip(tmp_path):
    samples = np.array([0, 1000, -1000, 32000, -32000], dtype="<i2")
    path = tmp_path / "rt.wav"
    save_mono_pcm16_wav(path, samples, 16000)
    loaded, sr = load_mono_pcm16_wav(path)
    assert sr == 16000
    np.testing.assert_array_equal(loaded, samples)


def test_resample_no_op_when_rates_equal():
    samples = np.array([1, 2, 3], dtype="<i2")
    out = resample_mono_pcm16(samples, 16000, 16000)
    np.testing.assert_array_equal(out, samples)


def test_resample_24k_to_16k_preserves_length_ratio():
    samples = np.zeros(2400, dtype="<i2")  # 100 ms at 24kHz
    out = resample_mono_pcm16(samples, 24000, 16000)
    assert abs(len(out) - 1600) <= 1


def test_resample_invalid_rate():
    with pytest.raises(ValueError):
        resample_mono_pcm16(np.array([1], dtype="<i2"), 0, 16000)


def test_pcm_f32_to_int16_clipping():
    buf = struct.pack("<3f", 0.0, 0.5, 2.0)  # 2.0 → clipped to +max
    out = pcm_f32_bytes_to_int16(buf)
    assert out[0] == 0
    assert out[1] == int(0.5 * 32768)
    assert out[2] == 32767


def test_pcm_f32_misaligned_buffer():
    with pytest.raises(ValueError):
        pcm_f32_bytes_to_int16(b"\x00\x00\x00")  # 3 bytes


def test_save_pcm_f32_as_wav(tmp_path):
    buf = struct.pack("<10f", *([0.1] * 10))
    out = tmp_path / "f.wav"
    save_pcm_f32_as_wav(out, buf, sample_rate=24000)
    loaded, sr = load_mono_pcm16_wav(out)
    assert sr == 24000
    assert len(loaded) == 10


# The server maps these to the same canonical output and emits identical bytes,
# so each must land in the matching WAV path (float32 vs int16).
@pytest.mark.parametrize("audio_format", ["pcm_f32", "pcm_f32le", "float32"])
def test_save_float_aliases(tmp_path, audio_format):
    buf = struct.pack("<10f", *([0.1] * 10))
    out = tmp_path / f"{audio_format}.wav"
    save_audio_as_wav(out, buf, 24000, audio_format)
    loaded, _ = load_mono_pcm16_wav(out)
    np.testing.assert_array_equal(loaded, pcm_f32_bytes_to_int16(buf))


@pytest.mark.parametrize("audio_format", ["pcm16", "pcm_16", "pcm_s16le"])
def test_save_pcm16_aliases(tmp_path, audio_format):
    buf = np.array([0, 1000, -1000, 32000, -32000], dtype="<i2").tobytes()
    out = tmp_path / f"{audio_format}.wav"
    save_audio_as_wav(out, buf, 16000, audio_format)
    loaded, _ = load_mono_pcm16_wav(out)
    np.testing.assert_array_equal(loaded, np.frombuffer(buf, dtype="<i2"))


def test_mulaw_expansion_matches_audioop():
    audioop = pytest.importorskip(
        "audioop", reason="audioop was removed in Python 3.13"
    )
    all_bytes = bytes(range(256))
    expected = np.frombuffer(audioop.ulaw2lin(all_bytes, 2), dtype="<i2")
    np.testing.assert_array_equal(mulaw_bytes_to_int16(all_bytes), expected)


def test_save_mulaw_as_wav_roundtrip(tmp_path):
    mulaw = bytes(range(256))
    out = tmp_path / "mulaw.wav"
    save_audio_as_wav(out, mulaw, 8000, "mulaw")
    loaded, sr = load_mono_pcm16_wav(out)
    assert sr == 8000
    assert len(loaded) == 256
    assert loaded.min() >= -32768 and loaded.max() <= 32767


@pytest.mark.parametrize("audio_format", ["mulaw", "ulaw", "twilio"])
def test_save_mulaw_aliases(tmp_path, audio_format):
    out = tmp_path / f"{audio_format}.wav"
    save_audio_as_wav(out, bytes(range(256)), 8000, audio_format)
    loaded, _ = load_mono_pcm16_wav(out)
    np.testing.assert_array_equal(loaded, mulaw_bytes_to_int16(bytes(range(256))))


def test_save_opus_as_wav_rejected(tmp_path):
    with pytest.raises(ValueError, match="Ogg container"):
        save_audio_as_wav(tmp_path / "x.wav", b"OggS...", 24000, "opus")


def test_save_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported audio_format"):
        save_audio_as_wav(tmp_path / "x.wav", b"\x00", 24000, "flac")  # type: ignore[arg-type]
