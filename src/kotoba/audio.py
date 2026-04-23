"""Audio loading, saving, and resampling helpers.

PCM16 helpers (load / save / resample) handle WAV / FLAC / OGG via
soundfile with a wave-module fallback. PCM float32 helpers are used by
the TTS path, since the TTS server emits `pcm_f32` frames.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised by import-time fallback
    sf = None  # type: ignore[assignment]


AudioFormat = Literal["pcm16", "pcm_f32"]


def load_mono_pcm16_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return mono int16 samples + sample rate.

    Supports WAV / FLAC / OGG / etc via soundfile, with a wave-module fallback
    for plain WAV when soundfile is unavailable.
    """

    path = Path(path)

    if sf is not None:
        try:
            audio, sample_rate = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.clip(audio * 32768.0, -32768.0, 32767.0).astype("<i2")
            return audio, int(sample_rate)
        except Exception:
            pass

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            raw = wf.readframes(frame_count)

        if sample_width != 2:
            raise ValueError(
                f"Only PCM16 WAV is supported by the wave fallback. "
                f"Got sample_width={sample_width} bytes"
            )

        audio = np.frombuffer(raw, dtype="<i2")
        if channels > 1:
            audio = audio.reshape(-1, channels)
            audio = np.mean(audio.astype(np.float32), axis=1).astype("<i2")

        return audio, sample_rate
    except wave.Error as exc:
        if sf is None:
            raise RuntimeError(
                f"Failed to load audio file {path}. "
                f"wave module error: {exc}. "
                f"Install soundfile (pip install soundfile) for non-WAV formats."
            ) from exc
        raise RuntimeError(
            f"Failed to load audio file {path}. Both soundfile and wave failed: {exc}"
        ) from exc


def save_mono_pcm16_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono int16 samples as a WAV file."""

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype("<i2").tobytes())


def resample_mono_pcm16(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample mono int16 audio with linear interpolation.

    Numpy-only so the SDK does not pull in a heavy DSP dependency.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return audio
    if len(audio) == 0:
        return audio

    src = audio.astype(np.float32)
    ratio = target_rate / source_rate
    target_len = max(1, int(round(len(src) * ratio)))
    src_x = np.arange(len(src), dtype=np.float32)
    dst_x = np.linspace(0, len(src) - 1, num=target_len, dtype=np.float32)
    dst = np.interp(dst_x, src_x, src)
    return np.clip(dst, -32768.0, 32767.0).astype("<i2")


def pcm_f32_bytes_to_int16(buf: bytes) -> np.ndarray:
    """Interpret little-endian float32 PCM bytes and clip-convert to int16."""

    if len(buf) == 0:
        return np.empty(0, dtype="<i2")
    if len(buf) % 4 != 0:
        raise ValueError(f"pcm_f32 buffer length {len(buf)} is not a multiple of 4")
    floats = np.frombuffer(buf, dtype="<f4")
    return np.clip(floats * 32768.0, -32768.0, 32767.0).astype("<i2")


def save_pcm_f32_as_wav(
    path: str | Path, buf: bytes, sample_rate: int = 24000
) -> None:
    """Write a float32-PCM byte buffer as a playable int16 WAV."""

    save_mono_pcm16_wav(path, pcm_f32_bytes_to_int16(buf), sample_rate)


def save_audio_as_wav(
    path: str | Path,
    buf: bytes,
    sample_rate: int,
    audio_format: AudioFormat,
) -> None:
    """Format-aware WAV writer used by the result models' `to_wav()` helpers."""

    if audio_format == "pcm_f32":
        save_pcm_f32_as_wav(path, buf, sample_rate)
        return
    if audio_format == "pcm16":
        if len(buf) % 2 != 0:
            raise ValueError(
                f"pcm16 buffer length {len(buf)} is not aligned to int16"
            )
        samples = np.frombuffer(buf, dtype="<i2").copy()
        save_mono_pcm16_wav(path, samples, sample_rate)
        return
    raise ValueError(f"unsupported audio_format: {audio_format!r}")
