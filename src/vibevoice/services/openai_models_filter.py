"""Heuristic filter for OpenAI /v1/models ids suitable for Chat Completions."""


def openai_model_id_for_chat_completions(model_id: str) -> bool:
    """Return True if ``model_id`` is likely usable with ``/v1/chat/completions``."""
    mid = model_id.strip().lower()
    if not mid:
        return False
    blocked = (
        "embedding",
        "embed",
        "whisper",
        "dall-e",
        "tts",
        "moderation",
        "davinci",
        "babbage",
        "curie",
        "ada:",
        "text-search",
        "code-search",
        "-embedding",
        "realtime",
        "audio",
    )
    if any(b in mid for b in blocked):
        return False
    if mid.startswith("gpt-"):
        return True
    if mid.startswith("ft:gpt-"):
        return True
    if mid.startswith("ft:") and ":gpt-" in mid:
        return True
    if mid.startswith("o1") or mid.startswith("o3") or mid.startswith("o4"):
        return True
    if mid.startswith("chatgpt-"):
        return True
    return False
