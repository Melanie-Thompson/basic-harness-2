from pydantic import BaseModel, Field

from .definition import ToolDefinition


class WordCounterInput(BaseModel):
    text: str = Field(description="Text to analyse")


def word_counter(text: str) -> str:
    """Count words, characters, and sentences in a block of text."""
    words = text.split()
    chars = len(text)
    sentences = sum(1 for c in text if c in ".!?")
    return f"{len(words)} words, {chars} characters, {sentences} sentences"


def _extractor(output: str):
    return [(output, "exact")]


TOOL = ToolDefinition(
    name="word_counter",
    description="Count words, characters, and sentences in a block of text.",
    inputSchema=WordCounterInput.model_json_schema(),
    fn=word_counter,
    extractor=_extractor,
)
