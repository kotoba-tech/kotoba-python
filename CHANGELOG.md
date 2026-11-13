# Changelog

## 0.3.3

- STT: `style_preference` takes `verbatim`. `{"verbatim": False}` removes
  fillers and restarts from the transcript (Japanese only), `True` keeps them,
  and leaving it unset keeps the deployment's default. 0.3.2 and earlier dropped
  the key without an error. `stream()` / `transcribe_stream()` send it as
  `session.input_audio_transcription.style_preference.verbatim`, and
  `transcribe(batch=True)` / `submit_job()` as the job's `verbatim` form field.
  With a `language` other than `"ja"` the streaming server answers
  `kotoba.ProtocolError` (`code="invalid_parameters"`,
  `param="session.input_audio_transcription"`).
- STT: `transcribe()` in its default mode (`POST /v1/speech-to-text`) raises
  `ValueError` for `style_preference` with `verbatim=False`: that endpoint has
  no such option and would return the fillers.
- STT docs: the streaming languages belong to the deployment, not to the SDK,
  which sends `language` as given. The README lists the codes fal
  `kotoba-streaming-stt` serves (`ja`, `en`, `zh`, `zh-tw`, `ko`, `yue`, `es`,
  `vi`) and says that, there, words spoken while two speakers overlap go to one
  of them.
- `scripts/live_check.py` (release check, not part of the package) takes
  `--language` and checks that a code the deployment does not serve raises
  `kotoba.ProtocolError` with
  `param="session.input_audio_transcription.target_language"`.

## 0.3.2

- STT: with `diarization=True` an empty `conversation.item.input_audio_transcription.completed`
  is now delivered as a `final_transcript` with `text == ""` (0.3.1 dropped it).
  It retires the provisional item of its `metadata["item_id"]`; without it a
  client kept that item's partial text forever and showed it twice once the
  server merged the text into another speaker turn. Without diarization empty
  completions are still dropped.
- STT docs: with `diarization=True`, partials are provisional. Each final is
  one speaker's turn and replaces the partials of its `metadata["item_id"]`
  (text and speaker may change), an empty final retires that item, and a final
  can arrive without partials. A whitespace-only partial has no `speaker_id`
  and no `metadata["item_id"]` (`None`); it continues the previous partial's
  item. The README and `examples/asr_stream_diarization_async.py` keep the
  transcript keyed by `item_id`.
- STT docs: an empty final carries `speaker_id` but no `start_ms` / `end_ms`
  (check `event.text` before counting finals per speaker), and a long session
  should drop the settled items it no longer displays. The `ProtocolError` row
  lists `code="too_many_concurrent_requests"` with `param="session.diarization"`
  (diarization service full) and
  `param="session.input_audio_transcription.target_language"` for a streaming
  `language` the deployment does not serve.
- STT docs: `turn_detection=False` with `diarization=True` is rejected by the
  server with `kotoba.ProtocolError` (`param="session.turn_detection"`); 0.3.1
  said it kept the provisional speakers only.

## 0.3.1

- STT: streaming speaker diarization. `stream()` / `transcribe_stream()` take
  `diarization=True` (sends `session.diarization: true`; deployments with a
  diarization service only). `StreamEvent.speaker_id` is the provisional
  speaker on `partial_transcript` (`None` on a whitespace-only partial) and the
  settled one on `final_transcript`
  (`speaker_1` … `speaker_8`, stable within one connection). Unless
  `turn_detection` is given the server applies `semantic_vad`. With
  `diarization=True`, `transcribe_stream()` yields those `StreamEvent`s instead
  of strings.
- STT: `diarization=True` combines with `keywords`; a deployment that cannot
  diarize the session raises `kotoba.ProtocolError` with
  `code="invalid_parameters"` and `param="session.diarization"` when the
  session starts, including a server without diarization support, whose
  acknowledgement does not enable it.
- `ProtocolError` has a `param` attribute: the error frame's `param` (the
  rejected field), or `None`.
- STT: `transcribe_stream()` takes `timestamps` like `stream()`.

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
- STT: `turn_detection` accepts `kotoba.SemanticVAD(eagerness=...)` (or a dict
  with `"type": "semantic_vad"`), which completes a turn soon after the model
  ends a sentence and waits longer inside an unfinished one.
