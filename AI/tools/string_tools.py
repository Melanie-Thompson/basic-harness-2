import hashlib
import base64

from pydantic import BaseModel, Field

from .definition import ToolDefinition


class HashStringInput(BaseModel):
    text: str = Field(description="Text to hash")
    algorithm: str = Field(default="sha256", description="Hash algorithm: md5, sha1, sha256, sha512")


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of the input text. Supports md5, sha1, sha256, sha512."""
    algo = algorithm.lower().strip()
    if algo not in ("md5", "sha1", "sha256", "sha512"):
        return f"error: unsupported algorithm '{algorithm}', use md5/sha1/sha256/sha512"
    h = hashlib.new(algo)
    h.update(text.encode())
    return f"{algo}:{h.hexdigest()}"


def _hash_extractor(output: str):
    return [(output, "exact")]


HASH_TOOL = ToolDefinition(
    name="hash_string",
    description="Compute a cryptographic hash of the input text. Supports md5, sha1, sha256, sha512.",
    inputSchema=HashStringInput.model_json_schema(),
    fn=hash_string,
    extractor=_hash_extractor,
)


class Base64EncodeInput(BaseModel):
    text: str = Field(description="Text to encode")


def base64_encode(text: str) -> str:
    """Base64-encode a string."""
    return base64.b64encode(text.encode()).decode()


def _base64_extractor(output: str):
    return [(output, "exact")]


BASE64_TOOL = ToolDefinition(
    name="base64_encode",
    description="Base64-encode a string.",
    inputSchema=Base64EncodeInput.model_json_schema(),
    fn=base64_encode,
    extractor=_base64_extractor,
)
