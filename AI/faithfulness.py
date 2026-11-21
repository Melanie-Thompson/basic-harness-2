"""Response faithfulness checker — BERT semantic similarity + per-tool extractors.

Three checks:
1. Relevance — is the tool output relevant to what the user asked?
2. Faithfulness — does the AI's answer match the tool output?
3. Extractors — do specific values from the tool output appear in the answer?

If the tool output isn't relevant to the question, the response is rejected
regardless of whether the answer faithfully reflects the tool output.
"""

import re
from datetime import timedelta

from dateutil import parser as dtparser

from AI.embeddings import semantic_similarity
from AI.models.message import FaithfulnessResult, ToolExecutionRecord
from AI.tools import TOOL_SCHEMAS, Extractor

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

SEMANTIC_THRESHOLD = 0.3
RELEVANCE_THRESHOLD = 0.35


def _check_numeric(value: str, ai_text: str) -> bool:
    try:
        expected = float(value.replace(",", ""))
    except ValueError:
        return False
    for match in _NUM_RE.finditer(ai_text):
        try:
            found = float(match.group().replace(",", ""))
        except ValueError:
            continue
        if abs(found - expected) <= max(1e-9, abs(expected) * 1e-6):
            return True
    return False


def _normalize_tz(dt):
    return dt.replace(tzinfo=None)


def _check_temporal(value: str, ai_text: str) -> bool:
    try:
        expected = _normalize_tz(dtparser.parse(value))
    except (ValueError, OverflowError):
        return False
    try:
        found_dt = _normalize_tz(dtparser.parse(ai_text, fuzzy_with_tokens=True)[0])
        if abs(found_dt - expected) < timedelta(minutes=2):
            return True
    except (ValueError, OverflowError):
        pass
    try:
        hm = expected.strftime("%H:%M")
        if hm in ai_text:
            return True
        h12 = expected.strftime("%-I:%M")
        if h12 in ai_text:
            return True
    except ValueError:
        pass
    return False


def _check_exact(value: str, ai_text: str) -> bool:
    return value.lower() in ai_text.lower()


_CHECKERS = {
    "exact": _check_exact,
    "numeric": _check_numeric,
    "temporal": _check_temporal,
}


import logging as _logging

_log = _logging.getLogger("faithfulness")


def check_relevance(
    user_message: str,
    tool_names: list[str],
    tool_descriptions: dict[str, str],
) -> tuple[bool, list[str]]:
    """Check if the selected tools are relevant to the user's question.

    Compares the user's question against each tool's description
    (what the tool does), not its output (which the LLM can game by
    choosing arguments that lexically overlap with the question).
    """
    if not tool_names or not user_message:
        return True, []

    mismatches = []
    for name in tool_names:
        desc = tool_descriptions.get(name, "")
        if not desc:
            mismatches.append(f"tool '{name}' has no description — cannot verify relevance")
            continue

        score = semantic_similarity(user_message, desc)
        _log.info("relevance: '%s' vs tool '%s' desc → %.3f (threshold %.2f)",
                   user_message[:60], name, score, RELEVANCE_THRESHOLD)
        if score < RELEVANCE_THRESHOLD:
            mismatches.append(
                f"tool '{name}' ({desc[:60]}) not relevant to question "
                f"(similarity {score:.3f} < {RELEVANCE_THRESHOLD})"
            )

    relevant = len(mismatches) == 0
    return relevant, mismatches


def check_faithfulness(
    ai_text: str,
    records: list[ToolExecutionRecord],
) -> dict[str | None, FaithfulnessResult]:
    """Returns a dict keyed by tool_call_id -> FaithfulnessResult."""
    results: dict[str | None, FaithfulnessResult] = {}

    for record in records:
        score = semantic_similarity(record.tool_output, ai_text)

        contained = True
        mismatches: list[str] = []

        schema = TOOL_SCHEMAS.get(record.tool_name)
        extractor: Extractor | None = schema.get("extractor") if schema else None

        if extractor is not None:
            for value, kind in extractor(record.tool_output):
                checker = _CHECKERS.get(kind, _check_exact)
                if not checker(value, ai_text):
                    contained = False
                    mismatches.append(f"{kind}:{value}")
        else:
            if score < SEMANTIC_THRESHOLD:
                contained = False
                mismatches.append(f"semantic:{score:.2f}<{SEMANTIC_THRESHOLD}")

        results[record.tool_call_id] = FaithfulnessResult(
            semantic_score=score,
            contained=contained,
            faithful=contained,
            mismatches=mismatches,
        )

    return results
