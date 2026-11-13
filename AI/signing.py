"""HMAC signing for tool execution records using itsdangerous."""

import json
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_serializer: URLSafeTimedSerializer | None = None


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        secret = os.getenv("TOOL_HMAC_SECRET") or os.urandom(32).hex()
        _serializer = URLSafeTimedSerializer(secret)
    return _serializer


def canonicalize_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def sign_execution(tool_name: str, canonical_args: str, output: str) -> str:
    s = _get_serializer()
    return s.dumps({"t": tool_name, "a": canonical_args, "o": output})


def verify_execution(signature: str, max_age: int = 300) -> dict:
    s = _get_serializer()
    return s.loads(signature, max_age=max_age)
