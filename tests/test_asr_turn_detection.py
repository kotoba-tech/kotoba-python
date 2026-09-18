"""SDK request serialization and event decoding, without transport substitutes."""

import json

import pytest
from kotoba import ASRClient, AsyncASRClient, ServerVAD
from kotoba._ws_asr import ASRSession, AsyncASRSession
from pydantic import ValidationError


@pytest.mark.parametrize("factory", [ASRClient, AsyncASRClient], ids=["sync", "async"])
@pytest.mark.parametrize(("options", "expected"), [
    pytest.param({}, None, id="omitted"),
    pytest.param({"turn_detection": None}, None, id="none-omits-wire-field"),
    pytest.param({"turn_detection": False}, False, id="disabled"),
    pytest.param({"turn_detection": {}}, {"type": "server_vad", "silence_duration_ms": 800}, id="dict-defaults"),
    pytest.param({"turn_detection": {"silence_duration_ms": 800}}, {"type": "server_vad", "silence_duration_ms": 800}, id="custom-dict"),
    pytest.param({"turn_detection": ServerVAD(silence_duration_ms=0)}, {"type": "server_vad", "silence_duration_ms": 0}, id="typed-zero"),
    pytest.param({"turn_detection": {"silence_duration_ms": None}}, {"type": "server_vad"}, id="null-duration-uses-server-default"),
])
def test_session_request_serialization(factory, options, expected):
    session = factory().stream(url="ws://unused", language="ja", **options)
    if isinstance(session, ASRSession):
        session = session._asr
    payload = json.loads(json.dumps(session._session_update()))
    assert payload["type"] == "transcription_session.update"
    if expected is None:
        assert "turn_detection" not in payload["session"]
    else:
        assert payload["session"]["turn_detection"] == expected
    assert payload["session"]["input_audio_transcription"]["target_language"] == "ja"


@pytest.mark.parametrize("config", [
    pytest.param(True, id="true"),
    pytest.param(0, id="numeric-false"),
    pytest.param({"type": "other"}, id="unsupported-type"),
    pytest.param({"type": None}, id="null-type"),
    pytest.param({"silence_duration_ms": -1}, id="negative"),
    pytest.param({"silence_duration_ms": 1.5}, id="fractional"),
    pytest.param({"silence_duration_ms": "400"}, id="string-duration"),
    pytest.param({"silence_duration_ms": True}, id="boolean-duration"),
])
def test_invalid_turn_configuration(config):
    with pytest.raises(ValidationError):
        AsyncASRSession("ws://unused", language="ja", turn_detection=config)


@pytest.mark.asyncio
async def test_multiple_turn_events_preserve_ids_and_text():
    session = AsyncASRSession("ws://unused", language="ja")
    for item_id, text in [("turn-a", "Hello"), ("turn-b", "こんにちは")]:
        for kind, field in [("delta", "delta"), ("completed", "transcript")]:
            await session._handle_text_frame({
                "type": f"conversation.item.input_audio_transcription.{kind}",
                "item_id": item_id, field: text,
            })
    await session._handle_text_frame({"type": "input_audio_buffer.committed"})
    events = [event async for event in session]
    assert [event.type for event in events] == [
        "partial_transcript", "final_transcript", "partial_transcript", "final_transcript",
        "committed", "done",
    ]
    assert [(event.text, event.metadata["item_id"]) for event in events[:4]] == [
        ("Hello", "turn-a"), ("Hello", "turn-a"), ("こんにちは", "turn-b"), ("こんにちは", "turn-b"),
    ]
