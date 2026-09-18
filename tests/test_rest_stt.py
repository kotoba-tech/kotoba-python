"""Wire contract of POST /v1/speech-to-text (kotoba/_rest_stt.py)."""

import pytest
from kotoba._rest_stt import SPEECH_TO_TEXT_PATH, SpeechToTextRequest, parse_speech_to_text
from kotoba.errors import ProtocolError
from kotoba.models import Segment


def test_request_defaults_omit_optional_fields():
    request = SpeechToTextRequest.build(
        language=None, keywords=None, style_preference=None, with_timestamps=False, file_format="other"
    )

    assert request.path == SPEECH_TO_TEXT_PATH == "/speech-to-text"
    assert request.form == {"file_format": "other", "with_timestamps": "false"}
    assert not hasattr(request, "headers")  # auth is the session's Authorization header only


def test_request_encodes_every_option():
    request = SpeechToTextRequest.build(
        language="ja",
        keywords=["Kotoba", "字幕"],
        style_preference={"human_name": "kana"},
        with_timestamps=True,
        file_format="pcm_s16le_16",
    )

    assert request.form == {
        "file_format": "pcm_s16le_16",
        "with_timestamps": "true",
        "language": "ja",
        "keywords": ["Kotoba", "字幕"],
        "kana": "true",
    }


@pytest.mark.parametrize("spacing_time", [None, 0.8])
def test_parse_maps_timed_words_to_segments_and_keeps_metadata(spacing_time):
    result = parse_speech_to_text(
        {
            "language_code": "ja",
            "language_probability": 1.0,
            "text": "こんにちは 世界",
            "audio_duration_secs": 1.2,
            "words": [
                {"text": "こんにちは", "type": "word", "start": 0.0, "end": 0.8},
                {"text": " ", "type": "spacing", "start": spacing_time, "end": spacing_time},
                {"text": "世界", "type": "word", "start": 0.9, "end": 1.2},
            ],
        }
    )

    assert result.text == "こんにちは 世界"
    assert result.segments == [Segment(text="こんにちは", start=0.0, end=0.8), Segment(text="世界", start=0.9, end=1.2)]
    assert result.metadata == {"language_code": "ja", "language_probability": 1.0, "audio_duration_secs": 1.2}


def test_parse_without_timed_words_has_no_segments():
    result = parse_speech_to_text({"language_code": "en", "language_probability": 0.9, "text": "hi", "words": []})

    assert result.segments is None
    assert result.metadata == {"language_code": "en", "language_probability": 0.9}


_VALID = {"language_code": "ja", "language_probability": 1.0, "text": "hi", "words": []}


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        pytest.param({**_VALID, "text": None}, "text", id="text-null"),
        pytest.param({k: v for k, v in _VALID.items() if k != "text"}, "text", id="text-missing"),
        pytest.param({**_VALID, "text": 123}, "text", id="text-not-a-string"),
        pytest.param({k: v for k, v in _VALID.items() if k != "language_code"}, "language_code", id="language-missing"),
        pytest.param({**_VALID, "words": "none"}, "words", id="words-not-a-list"),
        pytest.param({**_VALID, "words": [{"start": 0.0, "end": 0.5}]}, "words.0.text", id="word-without-text"),
        pytest.param(["not", "an", "object"], "<root>", id="not-an-object"),
    ],
)
def test_parse_rejects_bodies_off_the_server_contract(payload, field):
    """A 200 whose body breaks the contract is a protocol failure, never an empty transcript."""
    with pytest.raises(ProtocolError, match=f"Malformed /speech-to-text response: {field}:") as info:
        parse_speech_to_text(payload)

    assert info.value.status_code == 200
    assert info.value.payload  # the offending body travels with the error
