"""Tests for OpenAI-backed podcast script / segmentation paths."""

from unittest.mock import patch

import pytest

from vibevoice.services.podcast_generator import podcast_generator


def test_generate_script_from_article_openai_requires_api_key() -> None:
    with pytest.raises(ValueError, match="openai_api_key"):
        podcast_generator.generate_script_from_article(
            "Short article body for the episode.",
            "News",
            "10 min",
            ["VoiceOne"],
            narrator_speaker_index=1,
            llm_provider="openai",
            openai_api_key=None,
        )


def test_generate_script_segments_openai_requires_api_key() -> None:
    with pytest.raises(ValueError, match="openai_api_key"):
        podcast_generator.generate_script_segments(
            "Speaker 1: Hello.",
            llm_provider="openai",
            openai_api_key="",
            num_voices=1,
            genre="News",
            genre_style="News",
        )


@patch("vibevoice.services.openai_text_client._chat_message_content")
def test_generate_script_openai_returns_cleaned_script(mock_chat: object) -> None:
    mock_chat.return_value = "Speaker 1: Hello from the test."
    from vibevoice.services.openai_text_client import generate_script_openai

    out = generate_script_openai(
        "Article text " * 50,
        "News",
        "10 min",
        1,
        api_key="sk-test",
        model="gpt-4o-mini",
    )
    assert "Speaker 1" in out
    mock_chat.assert_called_once()


@patch("vibevoice.services.openai_text_client._chat_message_content")
def test_generate_script_segments_openai_parses_json(mock_chat: object) -> None:
    mock_chat.return_value = """[
      {"segment_type": "intro_music", "start_time_hint": 0, "duration_hint": 5, "energy_level": "high"},
      {"segment_type": "dialogue", "speaker": "Speaker 1", "text": "Hi", "start_time_hint": 5, "duration_hint": 2, "energy_level": "medium"},
      {"segment_type": "outro_music", "start_time_hint": 10, "duration_hint": 8, "energy_level": "low"}
    ]"""
    from vibevoice.services.openai_text_client import generate_script_segments_openai

    segs = generate_script_segments_openai(
        "Speaker 1: Hi.",
        api_key="sk-test",
        model="gpt-4o-mini",
        estimated_duration_seconds=120.0,
        num_voices=1,
        genre="News",
        genre_style="News",
    )
    assert len(segs) >= 1
    assert segs[0].get("segment_type") == "intro_music"
