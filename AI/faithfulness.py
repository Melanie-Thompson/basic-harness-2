"""Response faithfulness checker — BERT semantic similarity + per-tool extractors."""

import re
from datetime import timedelta

from dateutil import parser as dtparser

from AI.embeddings import semantic_similarity
from AI.models.message import FaithfulnessResult, ToolExecutionRecord
from AI.tools import TOOL_SCHEMAS, Extractor

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

SEMANTIC_THRESHOLD = 0.3


# ── Extractor checkers ──────────────────────────────────────────────────────


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


def _check_temporal(value: str, ai_text: str) -> bool:
    try:
        expected = dtparser.parse(value)
    except (ValueError, OverflowError):
        return False
    try:
        found_dt = dtparser.parse(ai_text, fuzzy_with_tokens=True)[0]
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


# ── Main entry point ────────────────────────────────────────────────────────


def check_faithfulness(
    ai_text: str,
    records: list[ToolExecutionRecord],
) -> list[FaithfulnessResult]:
    results: list[FaithfulnessResult] = []

    for record in records:
        score = semantic_similarity(record.tool_output, ai_text)

        schema = TOOL_SCHEMAS.get(record.tool_name)
        extractor: Extractor | None = schema.get("extractor") if schema else None

        contained = True
        mismatches: list[str] = []

        if extractor is not None:
            for value, kind in extractor(record.tool_output):
                checker = _CHECKERS.get(kind, _check_exact)
                if not checker(value, ai_text):
                    contained = False
                    mismatches.append(f"{kind}:{value}")

        # Extractor containment overrides a low semantic score (handles
        # multi-tool responses where BERT compares against the full text).
        # But a low semantic score with no extractor means the response is
        # off-topic entirely.
        if extractor is not None:
            faithful = contained
        else:
            faithful = score >= SEMANTIC_THRESHOLD

        results.append(FaithfulnessResult(
            tool_name=record.tool_name,
            semantic_score=score,
            contained=contained,
            faithful=faithful,
            mismatches=mismatches,
        ))

    return results
