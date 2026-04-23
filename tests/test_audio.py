import struct

import numpy as np
import pytest

from kotoba.audio import (
    load_mono_pcm16_wav,
    pcm_f32_bytes_to_int16,
    resample_mono_pcm16,
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
