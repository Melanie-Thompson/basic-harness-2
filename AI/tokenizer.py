"""Lazy-loaded Qwen tokenizer for accurate token counting."""

import os

from langchain_core.messages import BaseMessage

_tokenizer = None

MODEL_ID = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def count_messages(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += count_tokens(content)
    return total
