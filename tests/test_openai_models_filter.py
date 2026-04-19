"""Unit tests for OpenAI model id filter (no FastAPI import)."""

from vibevoice.services.openai_models_filter import openai_model_id_for_chat_completions


def test_accepts_chat_models() -> None:
    assert openai_model_id_for_chat_completions("gpt-4o-mini") is True
    assert openai_model_id_for_chat_completions("gpt-4-turbo") is True
    assert openai_model_id_for_chat_completions("o3-mini") is True
    assert openai_model_id_for_chat_completions("ft:gpt-4o-mini:org:xxx") is True


def test_rejects_non_chat() -> None:
    assert openai_model_id_for_chat_completions("text-embedding-3-small") is False
    assert openai_model_id_for_chat_completions("whisper-1") is False
    assert openai_model_id_for_chat_completions("tts-1") is False
