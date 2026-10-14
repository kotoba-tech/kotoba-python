"""Wire contract of POST /v1/speech-to-text (kotoba/_rest_stt.py)."""

import json
from pathlib import Path

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


def test_request_sends_diarization_only_when_asked():
    on = SpeechToTextRequest.build(
        language="ja", keywords=None, style_preference=None, with_timestamps=True, file_format="other", diarization=True
    )
    off = SpeechToTextRequest.build(
        language="ja", keywords=None, style_preference=None, with_timestamps=True, file_format="other", diarization=False
    )
    assert on.form["diarization"] == "true"
    assert "diarization" not in off.form


_TWO_SPEAKERS = {
    "language_code": "ja",
    "language_probability": 1.0,
    "text": "おはようございます。はい、わかりました。",
    "audio_duration_secs": 9.0,
    "words": [
        {"text": "おはよう", "type": "word", "start": 0.0, "end": 0.6, "speaker_id": "speaker_1"},
        {"text": "ございます。", "type": "word", "start": 0.6, "end": 1.4, "speaker_id": "speaker_1"},
        {"text": " ", "type": "spacing", "start": None, "end": None, "speaker_id": None},
        {"text": "はい、", "type": "word", "start": 4.8, "end": 5.2, "speaker_id": "speaker_2"},
        {"text": "わかりました。", "type": "word", "start": 5.2, "end": 6.1, "speaker_id": "speaker_2"},
    ],
}


def test_parse_groups_attributed_words_into_utterances():
    result = parse_speech_to_text(_TWO_SPEAKERS, diarization=True)

    assert [(t.speaker_id, t.text, t.start, t.end) for t in result.utterances] == [
        ("speaker_1", "おはようございます。", 0.0, 1.4),
        ("speaker_2", "はい、わかりました。", 4.8, 6.1),
    ]
    assert [s.speaker_id for s in result.segments] == ["speaker_1", "speaker_1", "speaker_2", "speaker_2"]


def test_parse_diarization_without_timestamps_keeps_utterances_and_no_segments():
    payload = {
        **_TWO_SPEAKERS,
        "words": [{**w, "start": None, "end": None} for w in _TWO_SPEAKERS["words"]],
    }
    result = parse_speech_to_text(payload, diarization=True)

    assert result.segments is None
    assert [(t.speaker_id, t.text, t.start, t.end) for t in result.utterances] == [
        ("speaker_1", "おはようございます。", None, None),
        ("speaker_2", "はい、わかりました。", None, None),
    ]


def test_parse_without_diarization_has_no_utterances():
    """with_timestamps only: the server answers speaker_id=None everywhere and nothing is grouped."""
    payload = {**_TWO_SPEAKERS, "words": [{**w, "speaker_id": None} for w in _TWO_SPEAKERS["words"]]}
    result = parse_speech_to_text(payload)

    assert result.utterances is None
    assert all(s.speaker_id is None for s in result.segments)


def test_parse_diarized_response_without_attribution_is_one_unattributed_utterance():
    payload = {**_TWO_SPEAKERS, "words": [{**w, "speaker_id": None} for w in _TWO_SPEAKERS["words"]]}
    result = parse_speech_to_text(payload, diarization=True)

    assert [(u.speaker_id, u.text, u.start, u.end) for u in result.utterances] == [
        (None, "おはようございます。 はい、わかりました。", 0.0, 6.1),
    ]


def _word(text, speaker, start=None, end=None, type="word"):
    return {"text": text, "type": type, "start": start, "end": end, "speaker_id": speaker}


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        pytest.param(
            [_word("A", "speaker_1", 0.0, 1.0), _word("B", None, 1.0, 2.0), _word("C", "speaker_2", 2.0, 3.0)],
            [("speaker_1", "A", 0.0, 1.0), (None, "B", 1.0, 2.0), ("speaker_2", "C", 2.0, 3.0)],
            id="unattributed-run-between-speakers",
        ),
        pytest.param(
            # The model emits the line break before a new turn under the new speaker.
            [_word("A", "speaker_1", 0.0, 0.5), _word("\n", "speaker_2", 0.9, 0.9, type="spacing"),
             _word("B", "speaker_2", 1.0, 1.5)],
            [("speaker_1", "A", 0.0, 0.5), ("speaker_2", "B", 1.0, 1.5)],
            id="spacing-before-a-turn-carries-the-next-speaker",
        ),
        pytest.param(
            [_word(" ", None, type="spacing"), _word("(laughs)", "speaker_1", 0.0, 0.5, type="audio_event"),
             _word("A", "speaker_1", 0.5, 1.0), _word(" ", "speaker_1", 1.0, 1.0, type="spacing")],
            [("speaker_1", "A", 0.5, 1.0)],
            id="leading-spacing-and-audio-events-dropped",
        ),
        pytest.param(
            [_word("A", "speaker_1"), _word(" ", "speaker_1", type="spacing"), _word("B", "speaker_1")],
            [("speaker_1", "A B", None, None)],
            id="untimed-words-keep-none-times",
        ),
        pytest.param([], None, id="empty-transcript"),
    ],
)
def test_parse_utterance_grouping(words, expected):
    payload = {**_TWO_SPEAKERS, "text": "".join(w["text"] for w in words), "words": words}
    result = parse_speech_to_text(payload, diarization=True)

    grouped = result.utterances and [(u.speaker_id, u.text, u.start, u.end) for u in result.utterances]
    assert grouped == expected


def test_parse_real_two_speaker_staging_response():
    """Answer of kotoba-stt--staging (diarized v1.4 model, image stt-20261014-5) to a synthesized two-speaker clip."""
    payload = json.loads((Path(__file__).parent / "data" / "speech_to_text_diarized_ja.json").read_text("utf-8"))
    result = parse_speech_to_text(payload, diarization=True)

    assert [(u.speaker_id, u.text, u.start, u.end) for u in result.utterances] == [
        ("speaker_1", "おはようございます。今日の会議は10時から始めましょう。", 0.0, 4.04),
        ("speaker_2", "はい、わかりました。資料は昨日メールで送っておきました。", 4.8, 9.64),
        ("speaker_1", "ありがとうございます。では、予算の話から始めます。", 10.16, 14.28),
        ("speaker_2", "来月の予算は前月より2割ほど増える見込みです。", 14.72, 18.7),
    ]
    assert result.text == payload["text"]
    assert len(result.segments) == 58  # the 61 items minus the three line-break spacings
    assert all(s.speaker_id in ("speaker_1", "speaker_2") for s in result.segments)
    assert result.metadata == {"language_code": "ja", "language_probability": 1.0, "audio_duration_secs": 19.2}
