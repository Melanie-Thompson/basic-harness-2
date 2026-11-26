"""Benchmark suite — measures false block rate, false pass rate, latency,
and tool correctness against a labelled test set.

Each test case specifies:
- prompt: what the user says
- expected_tool: which tool should be called (or None for deflection)
- expected_blocked: whether the request should be blocked
- expected_values: key values that must appear in the response

Metrics:
- false_block_rate: relevant questions incorrectly blocked
- false_pass_rate: irrelevant questions incorrectly answered
- tool_accuracy: correct tool selected
- faithfulness_rate: responses grounded in tool output
- mean_latency_ms: average response time
- token_efficiency: tokens consumed per resolved task
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("benchmark")


@dataclass
class TestCase:
    id: str
    prompt: str
    expected_tool: str | None  # None = should be blocked
    expected_blocked: bool
    expected_values: list[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class TestResult:
    case_id: str
    prompt: str
    expected_tool: str | None
    expected_blocked: bool
    actual_tool: str | None
    actual_blocked: bool
    response_text: str
    values_found: list[str] = field(default_factory=list)
    values_missing: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    passed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "expected_tool": self.expected_tool,
            "expected_blocked": self.expected_blocked,
            "actual_tool": self.actual_tool,
            "actual_blocked": self.actual_blocked,
            "response_text": self.response_text[:500],
            "values_found": self.values_found,
            "values_missing": self.values_missing,
            "latency_ms": round(self.latency_ms, 1),
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


@dataclass
class BenchmarkReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    false_blocks: int = 0
    false_passes: int = 0
    tool_correct: int = 0
    tool_total: int = 0
    faithful: int = 0
    faithful_total: int = 0
    total_latency_ms: float = 0.0
    results: list[TestResult] = field(default_factory=list)

    @property
    def false_block_rate(self) -> float:
        relevant = sum(1 for r in self.results if not r.expected_blocked)
        return self.false_blocks / relevant if relevant else 0.0

    @property
    def false_pass_rate(self) -> float:
        irrelevant = sum(1 for r in self.results if r.expected_blocked)
        return self.false_passes / irrelevant if irrelevant else 0.0

    @property
    def tool_accuracy(self) -> float:
        return self.tool_correct / self.tool_total if self.tool_total else 0.0

    @property
    def faithfulness_rate(self) -> float:
        return self.faithful / self.faithful_total if self.faithful_total else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "false_block_rate": round(self.false_block_rate, 3),
            "false_pass_rate": round(self.false_pass_rate, 3),
            "tool_accuracy": round(self.tool_accuracy, 3),
            "faithfulness_rate": round(self.faithfulness_rate, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "results": [r.to_dict() for r in self.results],
        }


# ── Default test set ───────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    # Relevant — should call weather_fetch and return data
    TestCase(
        id="weather_london",
        prompt="What's the weather in London?",
        expected_tool="weather_fetch",
        expected_blocked=False,
        expected_values=["14", "overcast"],
        category="relevant",
    ),
    TestCase(
        id="weather_tokyo",
        prompt="What is the weather like in Tokyo?",
        expected_tool="weather_fetch",
        expected_blocked=False,
        expected_values=["22", "sunny"],
        category="relevant",
    ),
    TestCase(
        id="weather_new_york",
        prompt="Tell me the weather in New York",
        expected_tool="weather_fetch",
        expected_blocked=False,
        expected_values=["18", "partly cloudy"],
        category="relevant",
    ),
    TestCase(
        id="weather_generic",
        prompt="What's the weather in Houston?",
        expected_tool="weather_fetch",
        expected_blocked=False,
        expected_values=["20", "fair"],
        category="relevant",
    ),
    TestCase(
        id="weather_texas",
        prompt="What is the weather in Texas?",
        expected_tool="weather_fetch",
        expected_blocked=False,
        category="relevant",
    ),

    # Irrelevant — should be blocked
    TestCase(
        id="irrelevant_compliment",
        prompt="Tell me I'm pretty",
        expected_tool=None,
        expected_blocked=True,
        category="irrelevant",
    ),
    TestCase(
        id="irrelevant_love",
        prompt="But don't you love me?",
        expected_tool=None,
        expected_blocked=True,
        category="irrelevant",
    ),
    TestCase(
        id="irrelevant_history",
        prompt="When was the Battle of Hastings?",
        expected_tool=None,
        expected_blocked=True,
        category="irrelevant",
    ),
    TestCase(
        id="irrelevant_math",
        prompt="What is 2 + 2?",
        expected_tool=None,
        expected_blocked=True,
        category="irrelevant",
    ),
    TestCase(
        id="irrelevant_joke",
        prompt="Tell me a joke about programming",
        expected_tool=None,
        expected_blocked=True,
        category="irrelevant",
    ),

    # Edge cases — semantically close but irrelevant
    TestCase(
        id="edge_duck",
        prompt="What is a duck?",
        expected_tool=None,
        expected_blocked=True,
        category="edge",
    ),
    TestCase(
        id="edge_climate",
        prompt="Why is the climate changing?",
        expected_tool=None,
        expected_blocked=True,
        category="edge",
    ),
]


async def run_benchmark(
    stream_reply_fn,
    test_cases: list[TestCase] | None = None,
    registry=None,
    session_id: str = "benchmark",
) -> BenchmarkReport:
    """Run the benchmark suite and return a report."""
    cases = test_cases or DEFAULT_TEST_CASES
    report = BenchmarkReport()

    for case in cases:
        log.info("benchmark: running case '%s': %s", case.id, case.prompt)
        report.total += 1

        actual_tool: str | None = None
        actual_blocked = False
        response_text = ""
        values_found: list[str] = []
        values_missing: list[str] = []

        start = time.monotonic()
        try:
            async for evt in stream_reply_fn(
                case.prompt,
                history=[],
                session_id=f"{session_id}_{case.id}",
                registry=registry,
            ):
                evt_type = evt.get("type", "")
                if evt_type == "tool_call":
                    actual_tool = evt.get("tool")
                elif evt_type == "tool_blocked":
                    actual_blocked = True
                    response_text = evt.get("reason", "")
                elif evt_type == "token":
                    response_text += evt.get("token", "")
        except Exception as e:
            log.warning("benchmark case '%s' failed: %s", case.id, e)
            response_text = f"error: {e}"

        latency_ms = (time.monotonic() - start) * 1000

        # Check expected values in response
        for val in case.expected_values:
            if val.lower() in response_text.lower():
                values_found.append(val)
            else:
                values_missing.append(val)

        # Determine pass/fail
        passed = True
        failure_reason = ""

        if case.expected_blocked and not actual_blocked:
            passed = False
            failure_reason = "should have been blocked but wasn't"
            report.false_passes += 1
        elif not case.expected_blocked and actual_blocked:
            passed = False
            failure_reason = "incorrectly blocked"
            report.false_blocks += 1
        elif case.expected_tool and actual_tool != case.expected_tool:
            passed = False
            failure_reason = f"expected tool '{case.expected_tool}', got '{actual_tool}'"
        elif values_missing:
            passed = False
            failure_reason = f"missing values: {values_missing}"

        # Track tool accuracy
        if case.expected_tool:
            report.tool_total += 1
            if actual_tool == case.expected_tool:
                report.tool_correct += 1

        # Track faithfulness (for non-blocked cases with expected values)
        if not case.expected_blocked and case.expected_values:
            report.faithful_total += 1
            if not values_missing:
                report.faithful += 1

        if passed:
            report.passed += 1
        else:
            report.failed += 1

        result = TestResult(
            case_id=case.id,
            prompt=case.prompt,
            expected_tool=case.expected_tool,
            expected_blocked=case.expected_blocked,
            actual_tool=actual_tool,
            actual_blocked=actual_blocked,
            response_text=response_text,
            values_found=values_found,
            values_missing=values_missing,
            latency_ms=latency_ms,
            passed=passed,
            failure_reason=failure_reason,
        )
        report.results.append(result)
        report.total_latency_ms += latency_ms

        log.info(
            "benchmark: case '%s' %s (%.0fms)%s",
            case.id,
            "PASS" if passed else "FAIL",
            latency_ms,
            f" — {failure_reason}" if failure_reason else "",
        )

    log.info(
        "benchmark complete: %d/%d passed, false_block=%.1f%%, false_pass=%.1f%%, "
        "tool_accuracy=%.1f%%, faithfulness=%.1f%%, mean_latency=%.0fms",
        report.passed, report.total,
        report.false_block_rate * 100,
        report.false_pass_rate * 100,
        report.tool_accuracy * 100,
        report.faithfulness_rate * 100,
        report.mean_latency_ms,
    )

    return report
