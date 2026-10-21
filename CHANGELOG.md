# Changelog

## 0.3.0

- STT: `keywords` are served by every Kotoba ASR deployment, the streaming
  `kotoba-streaming-stt` on fal included; `transcribe()` and `stream()` /
  `transcribe_stream()` take the same list. Blank terms are dropped
  (`kotoba.models.normalize_keywords`); the count and length limits are the
  deployment's and are enforced by the server.
- TTS: `voice_settings` (`kotoba.VoiceSettings`: `speed` / `volume`
  multipliers, `pitch` in semitones) sets the session default on `stream()`
  and overrides it per response on `synthesize()`.
- TTS: when `speaker_id` is omitted the server's profile default voice is used
  (the SDK no longer fills in `male-01`).
- TTS: `initial_text` sends the first request with the session setup, and
  `initial_request_in_upgrade=True` carries it in the `X-Kotoba-TTS-Init`
  Upgrade header; a missing acknowledgement raises a `ProtocolError` that
  names the header after 8 s.
- TTS: `with_timestamps` returns character alignments in `audio_chunk`
  metadata.
- TTS: a response stopped by the server's sequence limit ends as `done` with
  `finish_reason: "length"` (not an error); the audio already received is kept.
- TTS: `max_length` is deprecated, ignored by the server and no longer sent.
- WebSocket sessions share one TLS context.
- WebSocket: an unexpected server close reads its code and reason from the
  received close frame; the deprecated `ConnectionClosed.code` / `.reason`
  (websockets 13.1+) no longer emit a `DeprecationWarning`. The
  `WebSocket closed unexpectedly: code=… reason=…` message is unchanged.
