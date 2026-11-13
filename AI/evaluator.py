"""Inline tool-execution verifier — deterministic HMAC check, no LLM."""

from itsdangerous import BadSignature, SignatureExpired

from AI.models.message import ToolExecutionRecord
from AI.signing import canonicalize_args, verify_execution


class VerificationError(Exception):
    pass


def verify_tool_execution(record: ToolExecutionRecord) -> None:
    try:
        payload = verify_execution(record.signature)
    except SignatureExpired as e:
        raise VerificationError(f"signature expired for tool '{record.tool_name}'") from e
    except BadSignature as e:
        raise VerificationError(f"invalid signature for tool '{record.tool_name}'") from e

    canonical = canonicalize_args(record.tool_input)
    if (
        payload["t"] != record.tool_name
        or payload["a"] != canonical
        or payload["o"] != record.tool_output
    ):
        raise VerificationError(
            f"payload mismatch for tool '{record.tool_name}': "
            "signature was issued for a different call"
        )
